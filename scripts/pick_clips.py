"""Waehlt Clip-Kandidaten aus dem Transkript und den Chat-Spitzen aus (erste Runde).

Aufruf: python scripts/pick_clips.py data/latest.json clips.json

Ablauf:
1. Signale sammeln: laute Stellen (Audio), Chat-Spitzen und Zuschauer-Clips (chat_moments aus
   chat_peaks.py). Rund um jedes Signal entsteht eine Zone (+/- 45 Sekunden). Gemini bekommt nur die
   staerksten Zonen mit ihrem Transkript, nicht mehr das ganze Transkript. Gibt es gar keine
   Signale, bekommt Gemini wie frueher das ganze Transkript.
2. Gemini nennt Kandidaten und schreibt die ersten und letzten Woerter jedes Clips woertlich ab.
3. Hier wird nachgeprueft: Diese Woerter werden im Wort-Transkript gesucht, Start und Ende rasten
   auf die echten Wortgrenzen ein. Stimmt es nicht, gibt es Punktabzug.
4. Die Bewertung wird nachgerechnet: Lautstaerke, Chat-Spitzen und Sprechdichte geben Pluspunkte,
   Erzaehlen und Reden geben Abzug.
5. Wer unter MIN_FINAL liegt, fliegt raus. Es gibt KEINE Notloesung: Sind nur schwache Kandidaten
   da, ist die Liste leer (der Workflow meldet dann "Keine guten Clips gefunden").

Die zweite Runde (review_clips.py) schaut sich jeden Kandidaten danach als Video an.

Optional (Umgebungsvariablen):
  MAX_CLIPS         hoechstens so viele Kandidaten gehen in die zweite Runde, Standard 10
  MIN_FINAL         Mindestpunktzahl nach dem Nachrechnen, Standard 6.5
  EXTRA_BEFORE_SEC  Puffer VOR jedem Kandidaten fuer die zweite Runde, Standard 15 (der Chat
                    reagiert verzoegert, Gemini kuerzt einen langweiligen Anfang wieder)
  EXTRA_AFTER_SEC   Puffer NACH jedem Kandidaten, Standard 8
  MAX_ZONES         so viele Zonen gehen hoechstens an Gemini, Standard 40
  CHAT_MIN_RATIO    ab diesem Anstieg gilt eine Chat-Spitze als Beleg fuer Action, Standard 3
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
EXTRA_BEFORE = int(os.environ.get("EXTRA_BEFORE_SEC", "15"))
EXTRA_AFTER = int(os.environ.get("EXTRA_AFTER_SEC", os.environ.get("EXTRA_SEC", "8")))
MAX_ZONES = int(os.environ.get("MAX_ZONES", "40"))
CHAT_MIN_RATIO = float(os.environ.get("CHAT_MIN_RATIO", "3"))
ZONE_PAD = 45  # Sekunden Kontext um jedes Signal
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
chat = d.get("chat_moments", [])


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


def chat_best(start, end):
    """Staerkster Chat-Moment, dessen vermuteter Anlass im Zeitraum liegt (oder None)."""
    inside = [m for m in chat if m["event_start"] < end and m["event_end"] > start]
    return max(inside, key=lambda m: m["factor"]) if inside else None


def chat_label(m):
    return "ZUSCHAUER-CLIP" if m.get("type") == "clip" else "CHAT"


def density(start, end):
    """Transkript-Zeilen pro 10 Sekunden. Viele kurze Zeilen = schnelles Hin und Her."""
    n = sum(1 for s in segs if s["start"] < end and s["end"] > start)
    return n / max(1.0, end - start) * 10


def silent(start, end):
    """True, wenn im Zeitraum nichts gesprochen wird."""
    return not any(s["start"] < end and s["end"] > start for s in segs)


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


# ---------- Zonen: nur die Bereiche mit Signalen gehen an Gemini ----------
def build_zones():
    sig = []
    for w in loud_windows:
        if w["factor"] >= 2.0:
            sig.append((w["start"], w["end"], w["factor"] / 2.0, "LAUT", w["factor"]))
    for m in chat:
        sig.append((m["event_start"], m["event_end"], m["factor"] / CHAT_MIN_RATIO, chat_label(m), m["factor"]))
    sig.sort(key=lambda x: x[0])
    zones = []
    for s, e, w, label, val in sig:
        s, e = max(lo, s - ZONE_PAD), min(hi, e + ZONE_PAD)
        if e <= s:
            continue
        if zones and s <= zones[-1]["end"]:
            z = zones[-1]
            z["end"] = max(z["end"], e)
        else:
            z = {"start": s, "end": e, "w": [], "labels": {}}
            zones.append(z)
        z["w"].append(w)
        z["labels"][label] = max(z["labels"].get(label, 0), val)
    for z in zones:
        z["score"] = sum(sorted(z["w"], reverse=True)[:2]) + (0.3 if len(z["labels"]) > 1 else 0)
    zones.sort(key=lambda z: z["score"], reverse=True)
    zones = zones[:MAX_ZONES]
    zones.sort(key=lambda z: z["start"])
    return zones


# ---------- Gemini ----------
def line(x):
    tags = ""
    peak, _ = loudness(x["start"], x["end"])
    if peak >= 2.0:
        tags += f" [LAUT x{peak:.1f}]"
    cb = chat_best(x["start"], x["end"])
    if cb:
        tags += f" [{chat_label(cb)} x{cb['factor']:.1f}]"
    return f"[{int(x['start'])}-{int(math.ceil(x['end']))}]{tags} {x['text']}"


zones = build_zones()
if zones:
    parts = []
    for z in zones:
        body = [line(x) for x in segs if x["start"] < z["end"] and x["end"] > z["start"]]
        labels = ", ".join(f"{k} x{v:.1f}" for k, v in sorted(z["labels"].items()))
        parts.append(
            f"=== Zone [{int(z['start'])}-{int(math.ceil(z['end']))}] Signale: {labels} ===\n"
            + ("\n".join(body) if body else "(in dieser Zone wird kaum gesprochen)")
        )
    transcript = "\n\n".join(parts)
    print(f"Modus Zonen: {len(zones)} Zonen, {sum(z['end'] - z['start'] for z in zones) / 60:.0f} Minuten von {(hi - lo) / 60:.0f}")
    intro = (
        "Unten stehen nicht alle Zeilen des VODs, sondern nur die Zonen, in denen Signale auftauchten "
        "(laut, Chat-Ausbruch, Zuschauer-Clips). Jede Zone beginnt mit '=== Zone [Start-Ende] ... ==='. "
        "Kandidaten liegen innerhalb einer Zone."
    )
else:
    transcript = "\n".join(line(x) for x in segs)
    print("Modus Volltext: keine Signale gefunden, Gemini bekommt das ganze Transkript")
    intro = "Unten steht das Transkript des VOD-Ausschnitts."

N = MAX_CLIPS + 5  # etwas mehr nennen lassen, ein Teil faellt beim Nachrechnen raus

prompt = f"""Du bist strenger Editor fuer YouTube Shorts und TikToks aus Twitch-Streams und suchst Momente,
bei denen fremde Zuschauer haengen bleiben. {intro}
In eckigen Klammern stehen Start und Ende jeder Zeile in Sekunden im VOD, zum Beispiel [3605-3609].

Markierungen an den Zeilen:
- [LAUT x3.2]: dort war es deutlich lauter als sonst (x3.2 = dreimal so laut wie ueblich). Starker
  Hinweis auf Geschrei, Lachen, Schreck und Chaos.
- [CHAT x4.2]: dort schrieb der Chat ploetzlich viel mehr als sonst (x4.2 = Staerke des Anstiegs,
  inklusive Emotes wie LUL/KEKW und Spam). Der Chat reagiert mit 5 bis 15 Sekunden Verzoegerung. Die
  Markierung sitzt schon an der Stelle, an der der Anlass vermutlich passiert ist. Starte den Clip
  lieber etwas frueher als zu spaet, damit der Anlass noch drin ist.
- [ZUSCHAUER-CLIP x5.0]: Zuschauer haben genau dort selbst einen Twitch-Clip erstellt. Sehr starkes
  Zeichen, dass es ein Highlight war.
Je mehr Markierungen zusammenkommen (laut UND Chat), desto wahrscheinlicher ein guter Moment.
Aber auch eine Markierung allein ist keine Garantie: Pruefe, ob im Text wirklich etwas passiert.

SUCHE ACTION UND REAKTIONEN, KEINE ERZAEHLUNGEN.
Gut, in dieser Reihenfolge:
1. Chaos und Action im Spiel mit lauter Reaktion: Schreien, Lachen, Fluchen, Panik, Jagd, Falle,
   knappe Rettung, Kill, Fail, Troll, Jumpscare. Bei Manhunt zum Beispiel: gejagt werden,
   ausgetrickst werden, im letzten Moment entkommen.
2. Schnelles Hin und Her zwischen mehreren Leuten, Lachanfaelle, Streit, Ueberraschung.
3. Eine Pointe mit sofortiger Reaktion.
Typisch dafuer: viele kurze Zeilen, Ausrufe, Wiederholungen, Lachen, Markierungen.

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
  Transkript steht, und nimm die Zeilen, die zu start und end gehoeren. Wird am Anfang oder Ende
  nichts gesprochen, lass das Feld leer.
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

    first_words = str(c.get("first_words", "")).strip()
    a = locate(first_words, start)
    start_ok = bool(a and a[0] >= MIN_MATCH)
    if start_ok:
        start = max(lo, a[1] - LEAD)
    elif len(first_words.split()) < 2 and silent(start, start + 4):
        start_ok = True  # Anfang ohne Sprache: nichts zum Einrasten, aber auch kein Fehler
        notes.append("Anfang ohne Sprache")
    else:
        penalty += 1.0
        notes.append("Anfang nicht im Transkript gefunden")

    last_words = str(c.get("last_words", "")).strip()
    b = locate(last_words, end)
    if b and b[0] >= MIN_MATCH and b[2] > start:
        end = min(hi, b[2] + TAIL)
    elif len(last_words.split()) < 2 and silent(max(lo, end - 4), end):
        notes.append("Ende ohne Sprache")
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
    cb = chat_best(start, end)
    chat_peak = cb["factor"] if cb else 0.0
    dens = density(start, end)
    bonus = (
        min(2.0, max(0.0, (peak - 1.5) * 0.5))  # laut
        + min(1.0, n_loud * 0.25)  # lange laut
        + min(1.0, max(0.0, (dens - 2.5) * 0.4))  # viele kurze Zeilen
        + min(2.0, max(0.0, (chat_peak - 2.0) * 0.4))  # Chat-Ausbruch / Zuschauer-Clips
    )
    if kind in TALK_KINDS:
        penalty += 2.5
    if kind in ACTION_KINDS and peak < 1.8 and chat_peak < CHAT_MIN_RATIO:
        penalty += 1.0  # behauptet Action, aber weder laut noch im Chat etwas los

    cands.append(
        {
            "start": round(start, 1),
            "end": round(end, 1),
            "title": title,
            "kind": kind,
            "score": score,
            "reason": str(c.get("reason", "")),
            "loud_peak": peak,
            "chat_peak": round(chat_peak, 1),
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
        f"  {c['final']:5.2f}  Gemini {c['score']:.0f}, {c['kind']:<9} laut x{c['loud_peak']:.1f} chat x{c['chat_peak']:.1f}  "
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
    seg_s, seg_e = max(lo, s - EXTRA_BEFORE), min(hi, e + EXTRA_AFTER)
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
