"""Findet die Facecam in einem Clip (per Gemini mit Bildverstaendnis).

Aufruf: python scripts/detect_facecam.py clip.mp4 facecam.json [debug.jpg]

Schreibt facecam.json:
  {"found": true, "box": [x0, y0, x1, y1]}   (Anteile von 0 bis 1, oben links = 0,0)
  {"found": false, "reason": "..."}           (keine Facecam oder Fehler -> Clip wird ohne Facecam gebaut)

Dieses Skript bricht nie ab: Bei jedem Problem (auch wenn Gemini nicht erreichbar ist
oder das Limit erreicht ist) steht found=false in der Datei, und to_split.py
macht den Clip dann im normalen 9:16-Layout.
"""
import json
import os
import subprocess
import sys
import tempfile

from gemini import ask, image_part

video, out_path = sys.argv[1], sys.argv[2]
debug_path = sys.argv[3] if len(sys.argv) > 3 else None

PROMPT = """Das sind {n} Bilder aus demselben Twitch-Stream.
Finde die Facecam: das Live-Kamerabild des Streamers (eine Person vor der Kamera), das ins
Spielbild eingeblendet ist. Sie sitzt meistens in einer Ecke, oft mit Rahmen.

Gib EIN Rechteck zurueck, das die Facecam auf allen Bildern komplett umfasst, aber moeglichst
nichts vom Spiel. Koordinaten normiert auf 0 bis 1000 (0,0 = oben links, 1000,1000 = unten rechts).

Setze found auf false, wenn keine eingeblendete Facecam zu sehen ist (zum Beispiel nur Spiel,
oder die Kamera fuellt das ganze Bild).
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


def duration(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", path],
        check=True, capture_output=True, text=True,
    ).stdout
    return float(json.loads(out)["format"]["duration"])


def detect():
    """Gibt das Ergebnis als dict zurueck. Darf bei Problemen eine Ausnahme werfen."""
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

    # attempts=2: lieber schnell auf das normale Layout ausweichen als lange auf Gemini warten
    answer = ask(
        [{"text": PROMPT.format(n=len(frames))}] + [image_part(p) for p in frames],
        SCHEMA, temperature=0.1, attempts=2,
    )

    if not isinstance(answer, dict) or not answer.get("found"):
        return {"found": False, "reason": "Gemini sieht keine eingeblendete Facecam"}

    try:
        x0, y0, x1, y1 = (int(answer[k]) for k in ("xmin", "ymin", "xmax", "ymax"))
    except (KeyError, TypeError, ValueError):
        return {"found": False, "reason": "Koordinaten fehlen in der Antwort"}

    w, h = (x1 - x0) / 1000, (y1 - y0) / 1000
    # Plausibilitaet: nicht winzig, nicht fast das ganze Bild, nicht extrem schmal oder flach
    if not (0.06 <= w <= 0.6 and 0.06 <= h <= 0.7) or not (0.5 <= (w / h) * (16 / 9) <= 3):
        return {"found": False, "reason": f"Box unplausibel: {(x0, y0, x1, y1)}"}

    pad = 15  # etwas Rand, damit die Facecam nicht angeschnitten wird
    x0, y0, x1, y1 = max(0, x0 - pad), max(0, y0 - pad), min(1000, x1 + pad), min(1000, y1 + pad)
    box = [x0 / 1000, y0 / 1000, x1 / 1000, y1 / 1000]

    if debug_path:
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

    return {"found": True, "box": box}


try:
    result = detect()
except (Exception, SystemExit) as e:  # SystemExit kommt aus gemini.ask, wenn alle Modelle scheitern
    reason = f"Erkennung fehlgeschlagen: {e}"
    key = os.environ.get("GEMINI_API_KEY")
    if key:
        reason = reason.replace(key, "***")
    result = {"found": False, "reason": reason[:300]}

with open(out_path, "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False)
print("Facecam:", result)
