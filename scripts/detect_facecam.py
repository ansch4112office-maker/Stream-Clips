"""Findet die Facecam: zuerst kostenlos mit OpenCV (Gesichtserkennung ueber viele Bilder),
nur wenn das nichts Stabiles ergibt, mit Gemini als Rueckfall.

Aufruf: python scripts/detect_facecam.py facecam.json debug.jpg snip1.mp4 [snip2.mp4 ...]

Idee: Die Facecam ist statisch, Gesichter im Spiel wandern. Aus jedem Video werden mehrere Bilder
entnommen, in jedem werden Gesichter gesucht. Eine Gesichtsbox, die in den meisten Bildern an
fast derselben Stelle liegt, ist die Facecam. Die Box wird dann auf den ganzen Rahmen vergroessert
(Schultern, Rand) und, wenn sie nah an einer Bildkante liegt, bis zur Kante gezogen.

Schreibt facecam.json:
  {"found": true, "box": [x0, y0, x1, y1], "method": "opencv"|"gemini", ...}  (Anteile 0 bis 1)
  {"found": false, "reason": "..."}

Dieses Skript bricht nie ab: Bei jedem Problem steht found=false in der Datei.

Installieren: pip install opencv-python-headless numpy
"""
import json
import os
import statistics
import subprocess
import sys
import tempfile

try:
    import cv2
except ImportError:  # dann läuft nur der Gemini-Rückfall
    cv2 = None

out_path, debug_path = sys.argv[1], sys.argv[2]
videos = sys.argv[3:]

FRAC = (0.1, 0.3, 0.5, 0.7, 0.9)  # Positionen im Video, aus denen Bilder entnommen werden
MIN_AGREE_IOU = 0.5
MIN_SHARE = 0.5  # so viel Anteil der Bilder muss das Gesicht an derselben Stelle zeigen
MIN_HITS = 3
EDGE_SNAP = 0.08  # liegt die vergroesserte Box so nah an einer Kante, wird sie bis dahin gezogen


def duration(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", path],
        check=True, capture_output=True, text=True,
    ).stdout
    return float(json.loads(out)["format"]["duration"])


def extract_frames():
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
                     "-frames:v", "1", "-vf", "scale=960:-2", "-q:v", "3", p],
                    check=True,
                )
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


# ---------- Weg 1: OpenCV ----------
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


def expand(face):
    """Gesichtsbox -> Facecam-Rahmen (breiter, mit Schultern), an nahe Kanten gezogen."""
    fw, fh = face[2] - face[0], face[3] - face[1]
    cx = (face[0] + face[2]) / 2
    x0, x1 = cx - fw * 1.3, cx + fw * 1.3
    y0, y1 = face[1] - fh * 0.6, face[3] + fh * 0.9
    x0, y0, x1, y1 = max(0.0, x0), max(0.0, y0), min(1.0, x1), min(1.0, y1)
    if x0 < EDGE_SNAP:
        x0 = 0.0
    if y0 < EDGE_SNAP:
        y0 = 0.0
    if x1 > 1 - EDGE_SNAP:
        x1 = 1.0
    if y1 > 1 - EDGE_SNAP:
        y1 = 1.0
    return [x0, y0, x1, y1]


def detect_opencv(frames):
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

    face = median_box(best)
    box = expand(face)
    if not plausible(box):
        return None, f"Box unplausibel: {box}"
    return {"box": box, "hits": len(best), "frames": n}, None


# ---------- Weg 2: Gemini (nur Rueckfall) ----------
PROMPT = """Das sind {n} Bilder aus demselben Twitch-Stream.
Finde die Facecam: das Live-Kamerabild des Streamers (eine Person vor der Kamera), das ins
Spielbild eingeblendet ist. Sie sitzt meistens in einer Ecke, oft mit Rahmen.

Gib EIN Rechteck zurueck, das die Facecam auf allen Bildern komplett umfasst, aber moeglichst
nichts vom Spiel. Koordinaten normiert auf 0 bis 1000 (0,0 = oben links, 1000,1000 = unten rechts).

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


def detect_gemini(frames):
    if not os.environ.get("GEMINI_API_KEY"):
        return None, "kein GEMINI_API_KEY"
    from gemini import ask, image_part

    subsets = [frames]
    for sub in (frames[0::2], frames[1::2]):
        if sub and sub not in subsets:
            subsets.append(sub)
    boxes = []
    for sub in subsets:
        try:
            answer = ask([{"text": PROMPT.format(n=len(sub))}] + [image_part(p) for p in sub],
                         SCHEMA, temperature=0.1, attempts=2)
        except SystemExit as e:
            print("Gemini-Durchlauf fehlgeschlagen:", e)
            continue
        box = parse_gemini_box(answer)
        print("Gemini-Durchlauf:", box)
        if box:
            boxes.append(box)
    if len(boxes) < 2:
        return None, f"Gemini: nur {len(boxes)} von {len(subsets)} Durchlaeufen mit Facecam"
    med = median_box(boxes)
    agree = [b for b in boxes if iou(b, med) >= MIN_AGREE_IOU]
    if len(agree) < 2:
        return None, f"Gemini: Durchlaeufe widersprechen sich: {boxes}"
    x0, y0, x1, y1 = median_box(agree)
    pad = 0.015
    return {"box": [max(0.0, x0 - pad), max(0.0, y0 - pad), min(1.0, x1 + pad), min(1.0, y1 + pad)],
            "hits": len(agree), "frames": len(subsets)}, None


def draw_debug(frame, box):
    try:
        img = cv2.imread(frame)
        h, w = img.shape[:2]
        cv2.rectangle(img, (int(box[0] * w), int(box[1] * h)), (int(box[2] * w), int(box[3] * h)),
                      (0, 0, 255), 4)
        cv2.imwrite(debug_path, img)
    except Exception as e:
        print("Kontrollbild nicht erstellt:", e)


def detect():
    frames = extract_frames()
    if not frames:
        return {"found": False, "reason": "keine Bilder aus den Ausschnitten"}

    res, why = detect_opencv(frames)
    method = "opencv"
    if res is None:
        print("OpenCV hat nichts Stabiles gefunden:", why, "-> Gemini")
        res, why2 = detect_gemini(frames)
        method = "gemini"
        if res is None:
            return {"found": False, "reason": f"OpenCV: {why} | {why2}"}

    draw_debug(frames[0], res["box"])
    return {"found": True, "box": res["box"], "method": method,
            "hits": res["hits"], "frames": res["frames"]}


try:
    result = detect()
except (Exception, SystemExit) as e:
    reason = f"Erkennung fehlgeschlagen: {e}"
    key = os.environ.get("GEMINI_API_KEY")
    if key:
        reason = reason.replace(key, "***")
    result = {"found": False, "reason": reason[:300]}

with open(out_path, "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False)
print("Facecam:", result)
