"""Waehlt Clip-Kandidaten aus dem Transkript aus (erste Runde).

Aufruf: python scripts/pick_clips.py data/latest.json clips.json

Ablauf:
1. Gemini bekommt das Transkript mit Start und Ende jeder Zeile und markierten lauten Stellen.
2. Gemini nennt Kandidaten und schreibt die ersten und letzten Woerter jedes Clips woertlich ab.
3. Hier wird nachgeprueft: Diese Woerter werden im Wort-Transkript gesucht, Start und Ende rasten
   auf die echten Wortgrenzen ein. Stimmt es nicht, gibt es Punktabzug.
4. Die Bewertung wird nachgerechnet: Lautstaerke und Sprechdichte geben Pluspunkte, Erzaehlen und
   Reden geben Abzug.
5. Wer unter MIN_FINAL liegt, fliegt raus. Es gibt KEINE Notloesung mehr: Sind nur schwache
   Kandidaten da, ist die Liste leer (der Workflow meldet dann "Keine guten Clips gefunden").

Die zweite Runde (review_clips.py) schaut sich jeden Kandidaten danach als Video an.

Optional (Umgebungsvariablen):
  MAX_CLIPS   hoechstens so viele Kandidaten gehen in die zweite Runde, Standard 10
  MIN_FINAL   Mindestpunktzahl nach dem Nachrechnen, Standard 6.5
  EXTRA_SEC   Puffer vor und nach jedem Kandidaten fuer die zweite Runde, Standard 8
"""
import bisect
import difflib
import json
import math
import os
import re
import sys

from gemini import ask

src, dst = sys.argv[1], sys.argv[2]
MAX_CLIPS = int(os.environ.get("MAX_CLIPS", "10"))
MIN_FINAL = float(os.environ.get("MIN_FINAL", "6.5"))
EXTRA = int(os.environ.get("EXTRA_SEC", "8"))
MIN_LEN, MAX_LEN = 15, 45  # Sekunden
LEAD, TAIL = 0.5, 1.0  # Luft vor dem ersten und nach dem letzten Wort
MIN_MATCH = 0.6  # so aehnlich muessen abgeschriebene Woerter und Transkript mindestens sein
SEARCH = 25  # Sekunden um die Zeitangabe von Gemini herum wird nach den Woertern gesucht
TALK_KINDS = {"story", "gespraech"}  # Erzaehlen und Reden: wird abgewertet
ACTION_KINDS = {"action", "reaktion"}

with open(src, encoding="utf-8") as f:
    d = json.load(f)

offset = d.get("offset_sec", 0)
lo, hi = offset, offset + d["duration_sec"]
segs = [x for x in d["transcript"] if x["text"]]
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


def density(start, end):
    """Transkript-Zeilen pro 10 Sekunden. Viele kurze Zeilen = schnelles Hin und Her."""
    n = sum(1 for s in segs if s["start"] < end and s["end"] > start)
    return n / max(1.0, end - start) * 10


# ---------- Woerter mit Zeitstempel (fuer das Einrasten) ----------
def norm_word(w):
    return re.sub(r"\W+", "", w.lower())


def build_tokens():
    toks = []
    if d.get("words"):
        for w in d["words"]:
            n = norm_word(w["word"])
            if n:
                toks.append((n, w["start"], w["end"]))
    else:  # aeltere latest.json ohne Wortzeiten: Zeiten gleichmaessig auf die Woerter verteilen
        for s in segs:
            ws = [n for n in (norm_word(x) for x in s["text"].split()) if n]
            if not ws:
                continue
            step = (s["end"] - s["start"]) / len(ws)
            for i, n in enumerate(ws):
                toks.append((n, s["start"] + i * step, s["start"] + (i + 1) * step))
    return toks


tokens = build_tokens()
starts = [t[1] for t in tokens]


def locate(quote, around):
    """Sucht die abgeschriebenen Woerter nahe 'around'. Gibt (Aehnlichkeit, erste Zeit, letzte Zeit)
    zurueck oder None."""
    q = [n for n in (norm_word(w) for w in str(quote).split()) if n]
    if len(q) < 2 or not tokens:
        return None
    k, qs = len(q), " ".join(q)
    i0 = bisect.bisect_left(starts, around - SEARCH)
    i1 = bisect.bisect_right(starts, around + SEARCH)
    best = None
    for i in range(i0, min(i1, len(tokens) - k + 1)):
        s = " ".join(t[0] for t in tokens[i : i + k])
        r = difflib.SequenceMatcher(None, qs, s).ratio()
        dist = abs(tokens[i][1] - around)
        # bei (fast) gleich guten Treffern, etwa bei Wiederholungen wie "nein nein nein",
        # gewinnt der, der naeher an der Zeitangabe von Gemini liegt
        if best is None or r > best[0] + 0.02 or (r >= best[0] - 0.02 and dist < best[2]):
            best = (max(r, best[0]) if best else r, i, dist)
    if best is None:
        return None
    r, i = best[0], best[1]
    return r, tokens[i][1], tokens[i + k - 1][2]


# ---------- Gemini ----------
def line(x):
    peak, _ = loudness(x["start"], x["end"])
    tag = f" [LAUT x{peak:.1f}]" if peak >= 2.0 else ""
    return f"[{int(x['start'])}-{int(math.ceil(x['end']))}]{tag} {x['text']}"


transcript = "\n".join(line(x) for x in segs)
N = MAX_CLIPS + 5  # etwas mehr nennen lassen, ein Teil faellt beim Nachrechnen raus

prompt = f"""Du bist strenger Editor fuer YouTube Shorts und TikToks aus Twitch-Streams und suchst Momente,
bei denen fremde Zuschauer haengen bleiben. Unten steht das Transkript eines VOD-Ausschnitts.
In eckigen Klammern stehen Start und Ende jeder Zeile in Sekunden im VOD, zum Beispiel [3605-3609].
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
- Nenne bis zu {N} Kandidaten, die staerksten, nicht die ersten. Verteile sie ueber den ganzen
  Ausschnitt, wenn es an mehreren Stellen gute Momente gibt.
- WENN ES NICHTS WIRKLICH GUTES GIBT, GIB EINE LEERE LISTE ZURUECK. Das ist ausdruecklich erlaubt
  und besser als langweilige Clips. Fuelle nie auf.
- Laenge {MIN_LEN} bis {MAX_LEN} Sekunden. Klarer Einstieg in den ersten Sekunden, Ende kurz nach
  der Pointe, nicht mitten im Satz.
- start und end sind Sekunden im VOD, nur zwischen {lo} und {hi}. Kandidaten ueberschneiden sich nicht.
- first_words: die ersten 4 bis 8 Woerter, die im Clip gesprochen werden, WOERTLICH aus dem
  Transkript abgeschrieben. last_words: die letzten 4 bis 8 Woerter des Clips, ebenfalls woertlich.
  Damit wird ueberprueft, dass Zeit und Inhalt zusammenpassen. Schreib nur ab, was wirklich im
  Transkript steht, und nimm die Zeilen, die zu start und end gehoeren.
- title: deutscher Titel, hoechstens 8 Woerter, beschreibt genau das, was in diesem Zeitraum
  passiert. Kein Titel, der Vorwissen braucht, und keiner, der etwas verspricht, was nicht kommt.
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
                    "first_words": {"type": "STRING"},
                    "last_words": {"type": "STRING"},
                    "title": {"type": "STRING"},
                    "kind": {"type": "STRING"},
                    "score": {"type": "INTEGER"},
                    "reason": {"type": "STRING"},
                },
                "required": [
                    "start", "end", "first_words", "last_words",
                    "title", "kind", "score", "reason",
                ],
            },
        }
    },
    "required": ["clips"],
}

answer = ask([{"text": prompt}], schema)

# ---------- Kandidaten pruefen, einrasten, nachbewerten ----------
cands = []
for c in answer.get("clips", []):
    try:
        start, end, score = float(c["start"]), float(c["end"]), float(c["score"])
    except (KeyError, TypeError, ValueError):
        continue
    title = str(c.get("title", "clip"))
    start, end = max(lo, start), min(hi, end)
    if end <= start:
        continue

    penalty = 0.0
    notes = []

    a = locate(c.get("first_words", ""), start)
    start_ok = bool(a and a[0] >= MIN_MATCH)
    if start_ok:
        start = max(lo, a[1] - LEAD)
    else:
        penalty += 1.0
        notes.append("Anfang nicht im Transkript gefunden")

    b = locate(c.get("last_words", ""), end)
    if b and b[0] >= MIN_MATCH and b[2] > start:
        end = min(hi, b[2] + TAIL)
    else:
        penalty += 0.5
        notes.append("Ende nicht im Transkript gefunden")

    if end <= start:
        print("verworfen (Ende vor Anfang):", title)
        continue
    if end - start > MAX_LEN:  # zu lang: am letzten Satzende kappen, das noch reinpasst
        ends = [s["end"] for s in segs if start + MIN_LEN <= s["end"] <= start + MAX_LEN]
        end = min(max(ends) + 0.3, start + MAX_LEN) if ends else start + MAX_LEN
    if end - start < MIN_LEN:
        print("verworfen (zu kurz):", title)
        continue

    kind = str(c.get("kind", "")).strip().lower()
    peak, n_loud = loudness(start, end)
    dens = density(start, end)
    bonus = (
        min(2.0, max(0.0, (peak - 1.5) * 0.5))  # laut
        + min(1.0, n_loud * 0.25)  # lange laut
        + min(1.0, max(0.0, (dens - 2.5) * 0.4))  # viele kurze Zeilen
    )
    if kind in TALK_KINDS:
        penalty += 2.5
    if kind in ACTION_KINDS and peak < 1.8:  # behauptet Action, aber nirgends laut
        penalty += 1.0

    cands.append(
        {
            "start": round(start, 1),
            "end": round(end, 1),
            "title": title,
            "kind": kind,
            "score": score,
            "reason": str(c.get("reason", "")),
            "loud_peak": peak,
            "dichte": round(dens, 1),
            "start_geprueft": start_ok,
            "hinweise": notes,
            "final": round(score + bonus - penalty, 2),
        }
    )

cands.sort(key=lambda c: c["final"], reverse=True)

print("Kandidaten (beste zuerst):")
for c in cands:
    flag = "" if c["start_geprueft"] else "  (Anfang ungeprueft)"
    print(
        f"  {c['final']:5.2f}  Gemini {c['score']:.0f}, {c['kind']:<9} laut x{c['loud_peak']:.1f}  "
        f"{hms(c['start'])}-{hms(c['end'])}  {c['title']}{flag}"
    )

# Keine Notloesung: wer unter der Schwelle liegt, fliegt raus
good = [c for c in cands if c["final"] >= MIN_FINAL]
chosen = []
for c in good:
    if any(c["start"] < o["end"] and c["end"] > o["start"] for o in chosen):
        continue
    chosen.append(c)
    if len(chosen) == MAX_CLIPS:
        break

result = []
for i, c in enumerate(chosen, 1):
    s, e = int(c["start"]), int(math.ceil(c["end"]))
    seg_s, seg_e = max(lo, s - EXTRA), min(hi, e + EXTRA)
    result.append(
        {
            "name": f"{i}_{slug(c['title'])}",
            "title": c["title"],
            "kind": c["kind"],
            "punkte": c["final"],
            "reason": c["reason"],
            "start": hms(s),
            "end": hms(e),
            "start_sec": s,
            "end_sec": e,
            "seg_start": hms(seg_s),
            "seg_end": hms(seg_e),
            "seg_start_sec": seg_s,
        }
    )

with open(dst, "w", encoding="utf-8") as f:
    json.dump({"clips": result, "alle_kandidaten": cands}, f, ensure_ascii=False, indent=1)

if not result:
    print(f"Keine guten Clips gefunden (kein Kandidat mit mindestens {MIN_FINAL} Punkten).")
else:
    print("Weiter in die zweite Runde:")
    for c in result:
        print(f"  {c['name']}: {c['start']} bis {c['end']}  ({c['kind']}, {c['punkte']}) {c['reason']}")
