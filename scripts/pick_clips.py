"""Laesst Gemini aus dem Transkript die besten Clip-Momente aussuchen.

Aufruf: python scripts/pick_clips.py data/latest.json clips.json
Optional: CLIP_COUNT (Standard 5)

Ablauf:
1. Laute Stellen werden direkt im Transkript markiert ([LAUT x3.2]).
2. Gemini nennt mehr Kandidaten als noetig und bewertet jeden (Punkte 1-10 und Art).
3. Hier im Skript wird nachgerechnet: Lautstaerke gibt Pluspunkte, reine Erzaehlungen und
   Gespraeche Minuspunkte. Die besten ueberschneidungsfreien Kandidaten kommen in die Auswahl.
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
MIN_FINAL = 6.0  # darunter gilt ein Kandidat als zu schwach
TALK_KINDS = {"story", "gespraech"}  # Erzaehlen und Reden: wird abgewertet

with open(src, encoding="utf-8") as f:
    d = json.load(f)

offset = d.get("offset_sec", 0)
lo, hi = offset, offset + d["duration_sec"]
loud_windows = d["loud_moments"]


def hms(sec):
    sec = int(sec)
    return f"{sec // 3600:02d}:{sec % 3600 // 60:02d}:{sec % 60:02d}"


def slug(text):
    text = text.lower()
    for a, b in (("ä", "ae"), ("ö", "oe"), ("ü", "ue"), ("ß", "ss")):
        text = text.replace(a, b)
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_")[:40] or "clip"


def loudness(start, end):
    """Hoechster Lautstaerke-Faktor und Anzahl lauter 5-Sekunden-Fenster im Zeitraum."""
    inside = [w for w in loud_windows if w["start"] < end and w["end"] > start]
    if not inside:
        return 0.0, 0
    return max(w["factor"] for w in inside), len(inside)


def line(x):
    peak, _ = loudness(x["start"], x["end"])
    tag = f" [LAUT x{peak:.1f}]" if peak >= 2.0 else ""
    return f"[{int(x['start'])}]{tag} {x['text']}"


transcript = "\n".join(line(x) for x in d["transcript"] if x["text"])
N = COUNT * 3  # so viele Kandidaten verlangen wir von Gemini

prompt = f"""Du bist Editor fuer YouTube Shorts und TikToks aus Twitch-Streams und suchst Momente,
bei denen fremde Zuschauer haengen bleiben. Unten steht das Transkript eines VOD-Ausschnitts.
In eckigen Klammern steht die Startzeit jeder Zeile in Sekunden im VOD.
Zeilen mit [LAUT x3.2] sind Stellen, an denen es deutlich lauter war als sonst (x3.2 = dreimal
so laut wie ueblich). Das ist ein starker Hinweis auf Geschrei, Lachen, Schreck und Chaos.

SUCHE ACTION UND REAKTIONEN, KEINE ERZAEHLUNGEN.
Gut, in dieser Reihenfolge:
1. Chaos und Action im Spiel mit lauter Reaktion: Schreien, Lachen, Fluchen, Panik, Jagd, Falle,
   knappe Rettung, Kill, Fail, Troll, Jumpscare. Bei Manhunt zum Beispiel: gejagt werden,
   ausgetrickst werden, im letzten Moment entkommen.
2. Schnelles Hin und Her zwischen mehreren Leuten, Lachanfaelle, Streit, Ueberraschung.
3. Eine Pointe mit sofortiger Reaktion.
Typisch dafuer: viele kurze Zeilen, Ausrufe, Wiederholungen, Lachen, [LAUT]-Markierungen.

Schlecht, auch wenn es inhaltlich interessant klingt:
- Der Streamer erzaehlt, erklaert, redet ueber Themen, Smalltalk, Meinungen, Chat vorlesen
- lange ruhige Saetze ohne Reaktion, nichts passiert im Spiel
- Momente, die nur mit Vorwissen oder Kontext aus dem Stream funktionieren

Aufgabe:
- Nenne {N} Kandidaten. Verteile sie ueber den ganzen Ausschnitt, wenn es an mehreren Stellen gute
  Momente gibt. Nimm die staerksten, nicht die ersten.
- Laenge {MIN_LEN} bis {MAX_LEN} Sekunden, ideal 20 bis 45. Klarer Einstieg in den ersten Sekunden,
  Ende kurz nach der Pointe, nicht mitten im Satz.
- Zeiten in Sekunden, nur zwischen {lo} und {hi}. Kandidaten duerfen sich nicht ueberschneiden.
- kind: genau eines von action, reaktion, witz, story, gespraech.
  action = Geschehen im Spiel, reaktion = lautes Reagieren, witz = Pointe,
  story = der Streamer erzaehlt etwas, gespraech = ruhiges Reden.
- score: 1 bis 10, wie sehr wuerde ein Fremder beim Scrollen haengen bleiben. Streng sein:
  8 bis 10 nur fuer Momente, die wirklich knallen, ruhiges Reden hoechstens 4.
- reason: ein Satz, was passiert.

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
                    "kind": {"type": "STRING"},
                    "score": {"type": "INTEGER"},
                    "reason": {"type": "STRING"},
                },
                "required": ["start", "end", "title", "kind", "score", "reason"],
            },
        }
    },
    "required": ["clips"],
}

answer = ask([{"text": prompt}], schema)

# Kandidaten pruefen und selbst nachbewerten
cands = []
for c in answer.get("clips", []):
    try:
        start, end, score = int(c["start"]), int(c["end"]), float(c["score"])
    except (KeyError, TypeError, ValueError):
        continue
    start, end = max(lo, start), min(hi, end)
    if end - start > MAX_LEN:
        end = start + MAX_LEN
    if end - start < MIN_LEN:
        print("verworfen (zu kurz):", c.get("title"))
        continue
    kind = str(c.get("kind", "")).strip().lower()
    peak, n_loud = loudness(start, end)
    bonus = min(2.0, max(0.0, (peak - 1.5) * 0.5)) + min(1.0, n_loud * 0.25)
    penalty = 2.5 if kind in TALK_KINDS else 0.0
    cands.append(
        {
            "start": start,
            "end": end,
            "title": str(c.get("title", "clip")),
            "kind": kind,
            "score": score,
            "reason": str(c.get("reason", "")),
            "loud_peak": peak,
            "final": round(score + bonus - penalty, 2),
        }
    )

cands.sort(key=lambda c: c["final"], reverse=True)

print("Kandidaten (beste zuerst):")
for c in cands:
    print(f"  {c['final']:5.2f}  Gemini {c['score']:.0f}, {c['kind']:<9} laut x{c['loud_peak']:.1f}  "
          f"{hms(c['start'])}-{hms(c['end'])}  {c['title']}")

good = [c for c in cands if c["final"] >= MIN_FINAL]
if not good:
    print(f"Achtung: kein Kandidat erreicht {MIN_FINAL} Punkte, nehme die besten 3.")
    good = cands[:3]

clips = []
for c in good:
    if any(c["start"] < o["end"] and c["end"] > o["start"] for o in clips):
        continue
    clips.append(c)
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
            "kind": c["kind"],
            "punkte": c["final"],
            "reason": c["reason"],
        }
    )

with open(dst, "w", encoding="utf-8") as f:
    json.dump({"clips": result, "alle_kandidaten": cands}, f, ensure_ascii=False, indent=1)

print("Ausgewaehlt:")
for c in result:
    print(f"  {c['name']}: {c['start']} bis {c['end']}  ({c['kind']}, {c['punkte']}) {c['reason']}")
