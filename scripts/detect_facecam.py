"""Findet die Facecam (per Gemini mit Bildverstaendnis), mit mehreren Durchlaeufen und Abgleich.

Aufruf: python scripts/detect_facecam.py facecam.json debug.jpg snip1.mp4 [snip2.mp4 ...]

Aus jedem kurzen Video werden zwei Bilder entnommen. Gemini wird dreimal gefragt, jeweils mit einer
anderen Auswahl der Bilder. Gilt nur, wenn mindestens zwei Durchlaeufe eine plausible Box melden und
diese Boxen sich klar ueberlappen. Das Ergebnis ist der Median der uebereinstimmenden Boxen.
Zwei Durchlaeufe, die sich widersprechen, ergeben lieber "keine Facecam" (dann wird das normale
9:16-Layout benutzt) als eine falsche Box.

Schreibt facecam.json:
  {"found": true, "box": [x0, y0, x1, y1], "runs": 3, "agree": 3}   (Anteile 0 bis 1, oben links = 0,0)
  {"found": false, "reason": "..."}

Dieses Skript bricht nie ab: Bei jedem Problem steht found=false in der Datei.
"""
import json
import os
import statistics
import subprocess
import sys
import tempfile

from gemini import ask, image_part

out_path, debug_path = sys.argv[1], sys.argv[2]
videos = sys.argv[3:]

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

MIN_AGREE_IOU = 0.5


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
        for fi, frac in enumerate((0.3, 0.7)):
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


def parse_box(answer):
    """Gemini-Antwort -> Box (Anteile 0..1) oder None, wenn nichts oder Unplausibles gemeldet wurde."""
    if not isinstance(answer, dict) or not answer.get("found"):
        return None
    try:
        x0, y0, x1, y1 = (int(answer[k]) for k in ("xmin", "ymin", "xmax", "ymax"))
    except (KeyError, TypeError, ValueError):
        return None
    w, h = (x1 - x0) / 1000, (y1 - y0) / 1000
    # Plausibilitaet: nicht winzig, nicht fast das ganze Bild, nicht extrem schmal oder flach
    if not (0.06 <= w <= 0.6 and 0.06 <= h <= 0.7) or not (0.5 <= (w / h) * (16 / 9) <= 3):
        print("Box unplausibel:", (x0, y0, x1, y1))
        return None
    return [x0 / 1000, y0 / 1000, x1 / 1000, y1 / 1000]


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def median_box(boxes):
    return [statistics.median(b[i] for b in boxes) for i in range(4)]


def detect():
    frames = extract_frames()
    if not frames:
        return {"found": False, "reason": "keine Bilder aus den Ausschnitten"}

    # drei Durchlaeufe mit unterschiedlicher Bildauswahl
    subsets = [frames]
    for sub in (frames[0::2], frames[1::2]):
        if sub and sub not in subsets:
            subsets.append(sub)

    boxes = []
    for sub in subsets:
        try:
            # attempts=2: lieber schnell auf das normale Layout ausweichen als lange warten
            answer = ask(
                [{"text": PROMPT.format(n=len(sub))}] + [image_part(p) for p in sub],
                SCHEMA, temperature=0.1, attempts=2,
            )
        except SystemExit as e:
            print("Durchlauf fehlgeschlagen:", e)
            continue
        box = parse_box(answer)
        print("Durchlauf:", box)
        if box:
            boxes.append(box)

    runs = len(subsets)
    if len(boxes) < 2:
        return {"found": False, "reason": f"nur {len(boxes)} von {runs} Durchlaeufen haben eine Facecam gemeldet"}

    med = median_box(boxes)
    agree = [b for b in boxes if iou(b, med) >= MIN_AGREE_IOU]
    if len(agree) < 2:
        return {"found": False, "reason": f"Durchlaeufe widersprechen sich: {boxes}"}

    x0, y0, x1, y1 = median_box(agree)
    pad = 0.015  # etwas Rand, damit die Facecam nicht angeschnitten wird
    box = [max(0.0, x0 - pad), max(0.0, y0 - pad), min(1.0, x1 + pad), min(1.0, y1 + pad)]

    # Kontrollbild: erstes Bild mit rotem Rahmen um die erkannte Facecam
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-i", frames[0], "-vf",
             f"drawbox=x=iw*{box[0]}:y=ih*{box[1]}:w=iw*{box[2] - box[0]}:h=ih*{box[3] - box[1]}:color=red:t=4",
             "-frames:v", "1", debug_path],
            check=True,
        )
    except Exception as e:
        print("Kontrollbild nicht erstellt:", e)

    return {"found": True, "box": box, "runs": runs, "agree": len(agree)}


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
