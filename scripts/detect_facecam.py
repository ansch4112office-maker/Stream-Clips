
"""Findet die Facecam, und zwar das GANZE Kamerafenster, nicht nur das Gesicht.

Aufruf: python scripts/detect_facecam.py facecam.json debug.jpg snip1.mp4 [snip2.mp4 ...]

Ablauf:
  1. OpenCV sucht das Gesicht, das in den meisten Bildern an derselben Stelle liegt.
  2. Gemini sucht den Rahmen des ganzen Kamerafensters. Er zaehlt nur, wenn er zum Gesicht passt.
     Die Kanten werden danach auf feste Linien im Bild eingerastet.
  3. Nur wenn Gemini nichts Brauchbares liefert: grosszuegiger 16:9-Ausschnitt um das Gesicht.

Schreibt facecam.json:
  {"found": true, "box": [x0, y0, x1, y1], "method": "gemini"|"opencv", ...}  (Anteile 0 bis 1)
  {"found": false, "reason": "..."}

Dieses Skript bricht nie ab: Bei jedem Problem steht found=false in der Datei.

Installieren: pip install "opencv-python-headless<5" numpy
"""
import json
import os
import statistics
import subprocess
import sys
import tempfile

try:
    import cv2
    if not hasattr(cv2, "CascadeClassifier"):
        raise ImportError(f"OpenCV {cv2.__version__} hat keinen CascadeClassifier (Version <5 noetig)")
    CV2_ERROR = None
except ImportError as e:  # dann läuft nur Gemini
    cv2 = None
    CV2_ERROR = str(e)

FRAC = (0.1, 0.3, 0.5, 0.7, 0.9)  # Positionen im Video, aus denen Bilder entnommen werden
MIN_AGREE_IOU = 0.5
MIN_SHARE = 0.5  # so viel Anteil der Bilder muss das Gesicht an derselben Stelle zeigen
MIN_HITS = 3
EDGE_SNAP = 0.03  # liegt Geminis Rahmen so nah an einer Bildkante, wird er bis dahin gezogen
LINE_SEARCH = 0.02  # so weit darf eine Kante beim Einrasten wandern
LINE_SHARE = 0.6  # so viel der Kantenlaenge muss eine feste Linie sein
FACE_ZOOM_OUT = 4.2  # Rueckfall: Breite des Ausschnitts in Gesichtsbreiten (groesser = weiter weg)
HEADROOM = 0.25  # Rueckfall: Platz ueber dem Gesicht (Anteil der Ausschnitthoehe)


def duration(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", path],
        check=True, capture_output=True, text=True,
    ).stdout
    return float(json.loads(out)["format"]["duration"])


def extract_frames(videos):
    tmp = tempfile.mkdtemp()
    frames = []
    for vi, video in enumerate(videos):
        try:
            dur = duration(video)
        except Exception as e:
            print("Video uebersprungen:", video, e)
            continue
        for fi, frac in enumerate(FRAC):
            p = os.path.join(tmp, f"v{vi}_f{fi}.jpg")
            try:
                subprocess.run(
                    ["ffmpeg", "-y", "-v", "error", "-ss", f"{dur * frac:.2f}", "-i", video,
                     "-frames:v", "1", "-vf", "scale=1280:-2", "-q:v", "2", p],
                    check=True,
                )
                if os.path.exists(p):
                    frames.append(p)
            except Exception as e:
                print("Bild uebersprungen:", e)
    return frames


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def median_box(boxes):
    return [statistics.median(b[i] for b in boxes) for i in range(4)]


def plausible(box):
    w, h = box[2] - box[0], box[3] - box[1]
    return 0.06 <= w <= 0.6 and 0.06 <= h <= 0.7 and 0.5 <= (w / h) * (16 / 9) <= 3


# ---------- Gesicht mit OpenCV ----------
def find_faces(img, cascades):
    gray = cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    h, w = gray.shape
    found = []
    for c in cascades:
        rects = c.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6,
                                   minSize=(int(w * 0.03), int(w * 0.03)))
        for x, y, fw, fh in rects:
            found.append([x / w, y / h, (x + fw) / w, (y + fh) / h])
    return found


def detect_face(frames):
    if cv2 is None:
        return None, f"OpenCV nicht nutzbar: {CV2_ERROR}"
    cascades = []
    for name in ("haarcascade_frontalface_default.xml", "haarcascade_frontalface_alt2.xml"):
        c = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades, name))
        if not c.empty():
            cascades.append(c)
    if not cascades:
        return None, "keine Cascade-Dateien gefunden"

    per_frame = []
    for p in frames:
        img = cv2.imread(p)
        if img is not None:
            per_frame.append(find_faces(img, cascades))
    n = len(per_frame)
    if n < MIN_HITS:
        return None, f"nur {n} Bilder"

    # fuer jede gefundene Box zaehlen, in wie vielen Bildern etwas Aehnliches liegt
    best = None
    for boxes in per_frame:
        for b in boxes:
            members = []
            for other in per_frame:
                match = max((o for o in other if iou(b, o) >= MIN_AGREE_IOU),
                            key=lambda o: iou(b, o), default=None)
                if match:
                    members.append(match)
            if best is None or len(members) > len(best):
                best = members
    if not best or len(best) < max(MIN_HITS, MIN_SHARE * n):
        return None, f"kein stabiles Gesicht ({len(best or [])} von {n} Bildern)"
    return {"face": median_box(best), "hits": len(best), "frames": n}, None


def around_face(face):
    """Rueckfall: 16:9-Ausschnitt mit viel Luft. Bei 16:9-Quellen sind Breite und Hoehe als
    Anteil dann gleich gross. An Bildkanten wird verschoben statt abgeschnitten."""
    size = min(0.5, (face[2] - face[0]) * FACE_ZOOM_OUT)
    cx = (face[0] + face[2]) / 2
    x0 = min(max(0.0, cx - size / 2), 1.0 - size)
    y0 = min(max(0.0, face[1] - size * HEADROOM), 1.0 - size)
    return [x0, y0, x0 + size, y0 + size]


def fits_face(box, face):
    """Passt Geminis Rahmen zum Gesicht? Das Gesicht muss drin liegen und darf ihn nicht fuellen."""
    cx, cy = (face[0] + face[2]) / 2, (face[1] + face[3]) / 2
    inside = box[0] <= cx <= box[2] and box[1] <= cy <= box[3]
    return inside and (box[2] - box[0]) >= 1.8 * (face[2] - face[0])


# ---------- Kamerafenster mit Gemini ----------
PROMPT = """Das sind {n} Bilder aus demselben Twitch-Stream.
Finde die Facecam: das Live-Kamerabild des Streamers (eine Person vor der Kamera), das ins
Spielbild eingeblendet ist. Sie sitzt meistens in einer Ecke, oft mit Rahmen.

Gib das KOMPLETTE rechteckige Kamerafenster zurueck, von Aussenkante zu Aussenkante, also mit
Hintergrund, Schultern und allem, was die Kamera zeigt. NICHT nur Kopf oder Gesicht. Nimm aber
nichts vom Spiel oder von anderen Einblendungen mit. Koordinaten normiert auf 0 bis 1000
(0,0 = oben links, 1000,1000 = unten rechts).

Setze found auf false, wenn keine eingeblendete Facecam zu sehen ist (zum Beispiel nur Spiel,
oder die Kamera fuellt das ganze Bild). Eine Person, die im Spiel selbst vorkommt (Spielfigur,
Video im Spiel), ist keine Facecam.
"""

SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "found": {"type": "BOOLEAN"},
        "xmin": {"type": "INTEGER"},
        "ymin": {"type": "INTEGER"},
        "xmax": {"type": "INTEGER"},
        "ymax": {"type": "INTEGER"},
    },
    "required": ["found"],
}


def parse_gemini_box(answer):
    if not isinstance(answer, dict) or not answer.get("found"):
        return None
    try:
        x0, y0, x1, y1 = (int(answer[k]) for k in ("xmin", "ymin", "xmax", "ymax"))
    except (KeyError, TypeError, ValueError):
        return None
    box = [x0 / 1000, y0 / 1000, x1 / 1000, y1 / 1000]
    return box if plausible(box) else None


def detect_gemini(frames, face):
    if not os.environ.get("GEMINI_API_KEY"):
        return None, "kein GEMINI_API_KEY"
    from gemini import ask, image_part

    # drei verschiedene Bildgruppen mit hoechstens 6 Bildern, verteilt ueber alle Ausschnitte
    subsets = []
    for offset in range(3):
        sub = frames[offset::3][:6]
        if sub and sub not in subsets:
            subsets.append(sub)
    boxes = []
    for sub in subsets:
        try:
            answer = ask([{"text": PROMPT.format(n=len(sub))}] + [image_part(p) for p in sub],
                         SCHEMA, temperature=0.1, attempts=2)
        except (Exception, SystemExit) as e:
            print("Gemini-Durchlauf fehlgeschlagen:", str(e)[:200].replace(os.environ["GEMINI_API_KEY"], "***"))
            continue
        box = parse_gemini_box(answer)
        print("Gemini-Antwort:", str(answer)[:300])
        if box and face and not fits_face(box, face):
            print("Verworfen, passt nicht zum Gesicht (zu eng oder falsche Stelle):", box)
            box = None
        if box:
            boxes.append(box)
    if len(boxes) < 2:
        return None, f"Gemini: nur {len(boxes)} von {len(subsets)} Durchlaeufen brauchbar"
    med = median_box(boxes)
    agree = [b for b in boxes if iou(b, med) >= MIN_AGREE_IOU]
    if len(agree) < 2:
        return None, f"Gemini: Durchlaeufe widersprechen sich: {boxes}"
    return {"box": median_box(agree), "hits": len(agree), "frames": len(subsets)}, None


def snap_to_lines(frames, box):
    """Zieht jede Seite auf eine Linie, die in fast allen Bildern an derselben Stelle liegt
    (= Rand des Kamerafensters). Ohne klare Linie bleibt die Seite, wie sie ist."""
    if cv2 is None:
        return box
    import numpy as np

    sums, count, shape = [None, None], 0, None
    for p in frames:
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None or (shape and img.shape != shape):
            continue
        shape = img.shape
        for axis, (dx, dy) in enumerate(((1, 0), (0, 1))):
            mask = (np.abs(cv2.Sobel(img, cv2.CV_32F, dx, dy)) > 60).astype(np.float32)
            sums[axis] = mask if sums[axis] is None else sums[axis] + mask
        count += 1
    if count < MIN_HITS:
        return box
    h, w = shape
    stable_x, stable_y = (s / count >= 0.7 for s in sums)

    result = list(box)
    for i in range(4):
        vertical = i % 2 == 0  # linke und rechte Seite
        size = w if vertical else h
        if box[i] < EDGE_SNAP:
            result[i] = 0.0
            continue
        if box[i] > 1 - EDGE_SNAP:
            result[i] = 1.0
            continue
        center, reach = int(box[i] * size), int(LINE_SEARCH * size)
        if vertical:
            span = stable_x[int(box[1] * h):int(box[3] * h), :]
            scores = span.mean(axis=0)
        else:
            span = stable_y[:, int(box[0] * w):int(box[2] * w)]
            scores = span.mean(axis=1)
        lo, hi = max(0, center - reach), min(size, center + reach + 1)
        best = lo + int(np.argmax(scores[lo:hi]))
        if scores[best] >= LINE_SHARE:
            # 2 Pixel nach innen, damit der Rand selbst nicht im Bild ist
            result[i] = (best + (2 if i < 2 else -1)) / size
    return result


def draw_debug(frame, box, debug_path):
    try:
        img = cv2.imread(frame)
        h, w = img.shape[:2]
        cv2.rectangle(img, (int(box[0] * w), int(box[1] * h)), (int(box[2] * w), int(box[3] * h)),
                      (0, 0, 255), 4)
        cv2.imwrite(debug_path, img)
    except Exception as e:
        print("Kontrollbild nicht erstellt:", e)


def detect(videos, debug_path):
    frames = extract_frames(videos)
    if not frames:
        return {"found": False, "reason": "keine Bilder aus den Ausschnitten"}

    face_res, why_face = detect_face(frames)
    face = face_res["face"] if face_res else None
    print("Gesicht:", face if face else why_face)

    res, why = detect_gemini(frames, face)
    if res:
        method = "gemini"
        res["box"] = snap_to_lines(frames, res["box"])
    elif face:
        print("Kein brauchbarer Rahmen von Gemini:", why, "-> weiter Ausschnitt um das Gesicht")
        method = "opencv"
        res = {"box": around_face(face), "hits": face_res["hits"], "frames": face_res["frames"]}
    else:
        return {"found": False, "reason": f"OpenCV: {why_face} | {why}"}

    draw_debug(frames[0], res["box"], debug_path)
    return {"found": True, "box": res["box"], "method": method,
            "hits": res["hits"], "frames": res["frames"]}


def main():
    out_path, debug_path = sys.argv[1], sys.argv[2]
    try:
        result = detect(sys.argv[3:], debug_path)
    except (Exception, SystemExit) as e:
        reason = f"Erkennung fehlgeschlagen: {e}"
        key = os.environ.get("GEMINI_API_KEY")
        if key:
            reason = reason.replace(key, "***")
        result = {"found": False, "reason": reason[:300]}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False)
    print("Facecam:", result)


if __name__ == "__main__":
    main()
