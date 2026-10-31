"""Findet die Facecam in einem Clip (per Gemini mit Bildverstaendnis).

Aufruf: python scripts/detect_facecam.py clip.mp4 facecam.json [debug.jpg]

Schreibt facecam.json:
  {"found": true, "box": [x0, y0, x1, y1]}   (Anteile von 0 bis 1, oben links = 0,0)
  {"found": false}                            (keine Facecam gefunden)
"""
import json
import os
import subprocess
import sys
import tempfile

from gemini import ask, image_part

video, out_path = sys.argv[1], sys.argv[2]
debug_path = sys.argv[3] if len(sys.argv) > 3 else None


def duration(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", path],
        check=True, capture_output=True, text=True,
    ).stdout
    return float(json.loads(out)["format"]["duration"])


def save(result):
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f)
    print("Facecam:", result)


tmp = tempfile.mkdtemp()
dur = duration(video)
frames = []
for i, frac in enumerate((0.15, 0.4, 0.65, 0.9)):
    p = os.path.join(tmp, f"frame{i}.jpg")
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-ss", f"{dur * frac:.2f}", "-i", video,
         "-frames:v", "1", "-vf", "scale=960:-2", "-q:v", "3", p],
        check=True,
    )
    frames.append(p)

prompt = f"""Das sind {len(frames)} Bilder aus demselben Twitch-Stream.
Finde die Facecam: das Live-Kamerabild des Streamers (eine Person vor der Kamera), das ins
Spielbild eingeblendet ist. Sie sitzt meistens in einer Ecke, oft mit Rahmen.

Gib EIN Rechteck zurueck, das die Facecam auf allen Bildern komplett umfasst, aber moeglichst
nichts vom Spiel. Koordinaten normiert auf 0 bis 1000 (0,0 = oben links, 1000,1000 = unten rechts).

Setze found auf false, wenn keine eingeblendete Facecam zu sehen ist (zum Beispiel nur Spiel,
oder die Kamera fuellt das ganze Bild).
"""

schema = {
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

answer = ask([{"text": prompt}] + [image_part(p) for p in frames], schema, temperature=0.1)

if not answer.get("found"):
    save({"found": False})
    sys.exit(0)

try:
    x0, y0, x1, y1 = (int(answer[k]) for k in ("xmin", "ymin", "xmax", "ymax"))
except (KeyError, TypeError, ValueError):
    save({"found": False})
    sys.exit(0)

w, h = (x1 - x0) / 1000, (y1 - y0) / 1000
# Plausibilitaet: nicht winzig, nicht fast das ganze Bild, nicht extrem schmal oder flach
if not (0.06 <= w <= 0.6 and 0.06 <= h <= 0.7) or not (0.5 <= (w / h) * (16 / 9) <= 3):
    print("Box unplausibel, ignoriert:", (x0, y0, x1, y1))
    save({"found": False})
    sys.exit(0)

pad = 15  # etwas Rand, damit die Facecam nicht angeschnitten wird
x0, y0, x1, y1 = max(0, x0 - pad), max(0, y0 - pad), min(1000, x1 + pad), min(1000, y1 + pad)
box = [x0 / 1000, y0 / 1000, x1 / 1000, y1 / 1000]
save({"found": True, "box": box})

if debug_path:
    # Kontrollbild: erstes Bild mit rotem Rahmen um die erkannte Facecam
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", frames[0], "-vf",
         f"drawbox=x=iw*{box[0]}:y=ih*{box[1]}:w=iw*{box[2] - box[0]}:h=ih*{box[3] - box[1]}:color=red:t=4",
         "-frames:v", "1", debug_path],
        check=True,
    )
