"""Laesst Gemini aus dem Transkript die besten Clip-Momente aussuchen.

Aufruf: python scripts/pick_clips.py data/latest.json clips.json
Optional: CLIP_COUNT (Standard 5)
"""
import json
import os
import re
import sys

from gemini import ask

src, dst = sys.argv[1], sys.argv[2]
COUNT = int(os.environ.get("CLIP_COUNT", "5"))
MIN_LEN, MAX_LEN = 15, 75  # Sekunden
PAD = 1  # Sekunden Puffer vorne und hinten

with open(src, encoding="utf-8") as f:
    d = json.load(f)

offset = d.get("offset_sec", 0)
lo, hi = offset, offset + d["duration_sec"]


def hms(sec):
    sec = int(sec)
    return f"{sec // 3600:02d}:{sec % 3600 // 60:02d}:{sec % 60:02d}"


def slug(text):
    text = text.lower()
    for a, b in (("ä", "ae"), ("ö", "oe"), ("ü", "ue"), ("ß", "ss")):
        text = text.replace(a, b)
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_")[:40] or "clip"


transcript = "\n".join(f"[{int(x['start'])}] {x['text']}" for x in d["transcript"] if x["text"])
loud = "\n".join(
    f"{int(x['start'])}-{int(x['end'])}: {x['factor']}x lauter als normal"
    for x in d["loud_moments"][:25]
)

prompt = f"""Du bist Editor fuer YouTube Shorts und TikToks aus Twitch-Streams.
Unten steht das Transkript eines VOD-Ausschnitts. In eckigen Klammern steht die Startzeit
jeder Zeile in Sekunden im VOD. Suche die {COUNT} besten Momente fuer kurze Clips.

Gute Momente:
- witzig, ueberraschend oder emotional (Rage, Freude, Schreck, starke Reaktion)
- eine kleine Geschichte mit Pointe
- ohne Vorwissen verstaendlich, klarer Einstieg in den ersten Sekunden, Ende kurz nach der Pointe
- Laenge {MIN_LEN} bis {MAX_LEN} Sekunden, ideal 25 bis 45

Regeln:
- Zeiten in Sekunden, nur zwischen {lo} und {hi}
- Die Clips duerfen sich nicht ueberschneiden
- Nicht mitten im Satz anfangen oder aufhoeren
- Der beste Clip zuerst
- Nur Momente nehmen, die wirklich gut sind. Lieber weniger als Fuellmaterial.

Lautstaerke-Spitzen (Hinweis, kein Muss):
{loud}

Transkript:
{transcript}
"""

schema = {
    "type": "OBJECT",
    "properties": {
        "clips": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "start": {"type": "INTEGER"},
                    "end": {"type": "INTEGER"},
                    "title": {"type": "STRING"},
                    "reason": {"type": "STRING"},
                },
                "required": ["start", "end", "title", "reason"],
            },
        }
    },
    "required": ["clips"],
}

answer = ask([{"text": prompt}], schema)

# Vorschlaege pruefen: im Bereich, passende Laenge, keine Ueberschneidung
clips = []
for c in answer.get("clips", []):
    start, end = int(c["start"]), int(c["end"])
    start, end = max(lo, start), min(hi, end)
    if end - start > MAX_LEN:
        end = start + MAX_LEN
    if end - start < MIN_LEN:
        print("verworfen (zu kurz):", c.get("title"))
        continue
    if any(start < o["end"] and end > o["start"] for o in clips):
        print("verworfen (ueberschneidet sich):", c.get("title"))
        continue
    clips.append({"start": start, "end": end, "title": c["title"], "reason": c["reason"]})
    if len(clips) == COUNT:
        break

if not clips:
    raise SystemExit("Gemini hat keine brauchbaren Clips vorgeschlagen.")

result = []
for i, c in enumerate(clips, 1):
    result.append(
        {
            "name": f"{i}_{slug(c['title'])}",
            "start": hms(max(lo, c["start"] - PAD)),
            "end": hms(min(hi, c["end"] + PAD)),
            "title": c["title"],
            "reason": c["reason"],
        }
    )

with open(dst, "w", encoding="utf-8") as f:
    json.dump({"clips": result}, f, ensure_ascii=False, indent=1)

for c in result:
    print(f"{c['name']}: {c['start']} bis {c['end']}  ({c['reason']})")
