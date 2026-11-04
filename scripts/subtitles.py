"""Baut aus den Wortzeiten eine Untertitel-Datei (.ass) im Shorts-Stil: 2 bis 3 Woerter pro Einblendung,
gross, weiss mit dickem schwarzem Rand.

Aufruf: python scripts/subtitles.py data/latest.json review.json subs.ass

Die Zeiten werden auf den fertig zugeschnittenen Clip umgerechnet (Start = review.json start_abs).
Gibt es keine Wortzeiten oder keine Woerter im Clip, wird keine Datei geschrieben (der Clip bekommt
dann einfach keine Untertitel).

Einstellungen ueber Umgebungsvariablen (alle optional):
  SUB_Y       senkrechte Position der Untertitel im 1080x1920-Bild, Standard 1250
  SUB_SIZE    Schriftgroesse, Standard 88
  SUB_WORDS   hoechstens so viele Woerter pro Einblendung, Standard 3
"""
import json
import os
import re
import sys

latest, review, out = sys.argv[1:4]

SUB_Y = int(os.environ.get("SUB_Y", "1250"))
SIZE = int(os.environ.get("SUB_SIZE", "88"))
MAX_WORDS = int(os.environ.get("SUB_WORDS", "3"))
PAUSE = 0.6  # ab dieser Pause (Sekunden) beginnt eine neue Einblendung
MAX_CHARS = 22  # laengere Einblendungen passen nicht mehr in die Bildbreite

with open(latest, encoding="utf-8") as f:
    d = json.load(f)
with open(review, encoding="utf-8") as f:
    r = json.load(f)

t0, t1 = r["start_abs"], r["end_abs"]
words = [w for w in d.get("words", []) if w["end"] > t0 and w["start"] < t1 and w["word"].strip()]
if not words:
    print("Keine Wortzeiten im Clip, keine Untertitel.")
    sys.exit(0)


def clean(w):
    w = w.replace("{", "").replace("}", "").replace("\\", "")
    return w.strip().upper()


# Woerter zu kurzen Gruppen zusammenfassen
groups, cur = [], []
for w in words:
    text = clean(w["word"])
    if not text:
        continue
    if cur:
        gap = w["start"] - cur[-1]["end"]
        joined = " ".join(x["text"] for x in cur) + " " + text
        ends_sentence = re.search(r"[.!?]$", cur[-1]["text"]) is not None
        if gap > PAUSE or len(cur) >= MAX_WORDS or len(joined) > MAX_CHARS or ends_sentence:
            groups.append(cur)
            cur = []
    cur.append({"text": text, "start": w["start"], "end": w["end"]})
if cur:
    groups.append(cur)


def ts(sec):
    sec = max(0.0, sec)
    cs = int(round(sec * 100))
    return f"{cs // 360000}:{cs // 6000 % 60:02d}:{cs // 100 % 60:02d}.{cs % 100:02d}"


header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,DejaVu Sans,{SIZE},&H00FFFFFF,&H000000FF,&H00000000,&H64000000,-1,0,0,0,100,100,0,0,1,7,2,5,60,60,0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

lines = [header]
for i, g in enumerate(groups):
    start = g[0]["start"] - t0
    end = g[-1]["end"] - t0 + 0.25
    if i + 1 < len(groups):  # nicht in die naechste Einblendung hineinragen
        end = min(end, groups[i + 1][0]["start"] - t0)
    end = min(end, t1 - t0)
    if end - start < 0.15:
        end = start + 0.15
    text = " ".join(x["text"] for x in g)
    lines.append(f"Dialogue: 0,{ts(start)},{ts(end)},Default,,0,0,0,,{{\\pos(540,{SUB_Y})}}{text}\n")

with open(out, "w", encoding="utf-8") as f:
    f.write("".join(lines))
print(f"Untertitel: {len(groups)} Einblendungen aus {len(words)} Woertern -> {out}")
