"""Baut aus den Wortzeiten eine Untertitel-Datei (.ass) im Twitch-/Shorts-Stil: 2 bis 3 Woerter pro
Einblendung, gross, weiss mit dickem schwarzem Rand. Das gerade gesprochene Wort leuchtet in
Akzentfarbe (Standard gelb) und poppt kurz auf.

Aufruf: python scripts/subtitles.py data/latest.json review.json subs.ass

Die Zeiten werden auf den fertig zugeschnittenen Clip umgerechnet (Start = review.json start_abs).
Gibt es keine Wortzeiten oder keine Woerter im Clip, wird keine Datei geschrieben (der Clip bekommt
dann einfach keine Untertitel).

Einstellungen ueber Umgebungsvariablen (alle optional):
  SUB_Y       senkrechte Position der Untertitel im 1080x1920-Bild, Standard 1250
  SUB_SIZE    Schriftgroesse, Standard 88
  SUB_WORDS   hoechstens so viele Woerter pro Einblendung, Standard 3
  SUB_HL      Farbe des gerade gesprochenen Wortes als RRGGBB, Standard FFE100 (gelb),
              z. B. 00FF5A fuer gruen
  SUB_POP     Pop-Groesse in Prozent beim Einblenden eines Wortes, Standard 118 (100 = aus)
"""
import json
import os
import re
import sys

words_json, out = sys.argv[1:3]

SUB_Y = int(os.environ.get("SUB_Y", "1250"))
SIZE = int(os.environ.get("SUB_SIZE", "88"))
MAX_WORDS = int(os.environ.get("SUB_WORDS", "3"))
HL = os.environ.get("SUB_HL", "FFE100").strip().lstrip("#")
POP = int(os.environ.get("SUB_POP", "118"))
PAUSE = 0.6  # ab dieser Pause (Sekunden) beginnt eine neue Einblendung
MAX_CHARS = 22  # laengere Einblendungen passen nicht mehr in die Bildbreite

if not re.fullmatch(r"[0-9A-Fa-f]{6}", HL):
    HL = "FFE100"
# ASS nutzt BGR statt RGB
HL_ASS = f"&H{HL[4:6]}{HL[2:4]}{HL[0:2]}&"

with open(words_json, encoding="utf-8") as f:
    d = json.load(f)

t0, t1 = 0.0, float(d.get("duration", 1e9))
words = [w for w in d.get("words", []) if w["word"].strip()]
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


def render(g, active):
    """Text der Gruppe, Wort 'active' leuchtet und poppt kurz auf, der Rest bleibt weiss."""
    parts = []
    for k, x in enumerate(g):
        if k == active:
            pop = (
                f"\\fscx{POP}\\fscy{POP}\\t(0,120,\\fscx100\\fscy100)" if POP != 100 else ""
            )
            parts.append(f"{{\\1c{HL_ASS}{pop}}}{x['text']}{{\\r}}")
        else:
            parts.append(x["text"])
    return " ".join(parts)


lines = [header]
n_events = 0
for i, g in enumerate(groups):
    group_end = g[-1]["end"] - t0 + 0.25
    if i + 1 < len(groups):  # nicht in die naechste Einblendung hineinragen
        group_end = min(group_end, groups[i + 1][0]["start"] - t0)
    group_end = min(group_end, t1 - t0)

    for k, x in enumerate(g):
        start = x["start"] - t0
        # Wort leuchtet bis das naechste beginnt, das letzte bis zum Ende der Einblendung
        end = g[k + 1]["start"] - t0 if k + 1 < len(g) else group_end
        if k == 0:
            start = max(start, 0.0)
        if end - start < 0.08:
            end = start + 0.08
        text = render(g, k)
        lines.append(
            f"Dialogue: 0,{ts(start)},{ts(end)},Default,,0,0,0,,{{\\pos(540,{SUB_Y})}}{text}\n"
        )
        n_events += 1

with open(out, "w", encoding="utf-8") as f:
    f.write("".join(lines))
print(f"Untertitel: {len(groups)} Einblendungen ({n_events} Wort-Schritte) aus {len(words)} Woertern -> {out}")
