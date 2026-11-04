"""Zweite Runde: Gemini schaut sich den Kandidaten als Video an, entscheidet ueber Behalten oder
Aussortieren, korrigiert den Titel und legt Anfang und Ende (15 bis 45 Sekunden) endgueltig fest.

Aufruf: python scripts/review_clips.py raw.mp4 data/latest.json review.json trimmed.mp4

raw.mp4 ist der Ausschnitt mit Puffer um den Kandidaten (siehe pick_clips.py). Die Zeiten kommen
ueber Umgebungsvariablen, weil der Workflow sie aus der Matrix hat:
  SEG_START_SEC    Start von raw.mp4 im VOD (Sekunden)
  CLIP_START_SEC   vorgeschlagener Start im VOD
  CLIP_END_SEC     vorgeschlagenes Ende im VOD
  CLIP_TITLE       vorgeschlagener Titel
  CLIP_KIND        vorgeschlagene Art
  REVIEW_MIN       Mindestpunktzahl fuer "behalten", Standard 7

Schreibt review.json und (nur wenn der Clip behalten wird) trimmed.mp4.
Ist Gemini nicht erreichbar, wird der Clip mit den urspruenglichen Zeiten behalten und in
review.json als "geprueft": false markiert. Das Skript bricht deswegen nicht ab.
"""
import json
import os
import subprocess
import sys
import tempfile

from gemini import ask, video_part

raw, latest, out_json, out_mp4 = sys.argv[1:5]

SEG_START = float(os.environ["SEG_START_SEC"])
CLIP_START = float(os.environ["CLIP_START_SEC"])
CLIP_END = float(os.environ["CLIP_END_SEC"])
TITLE = os.environ.get("CLIP_TITLE", "clip")
KIND = os.environ.get("CLIP_KIND", "")
MIN_SCORE = float(os.environ.get("REVIEW_MIN", "7"))
MIN_LEN, MAX_LEN = 15.0, 45.0
TALK_KINDS = {"story", "gespraech"}


def probe_duration(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", path],
        check=True, capture_output=True, text=True,
    ).stdout
    return float(json.loads(out)["format"]["duration"])


def fit_length(s, e, dur):
    """Bringt [s, e] in 15 bis 45 Sekunden und innerhalb von 0..dur. None, wenn es nicht geht."""
    s, e = max(0.0, s), min(dur, e)
    if e - s > MAX_LEN:
        e = s + MAX_LEN
    if e - s < MIN_LEN:
        e = min(dur, s + MIN_LEN)
        if e - s < MIN_LEN:
            s = max(0.0, e - MIN_LEN)
    if e - s < MIN_LEN - 0.01:
        return None
    return s, e


dur = probe_duration(raw)
plan_s, plan_e = CLIP_START - SEG_START, CLIP_END - SEG_START
plan = fit_length(plan_s, plan_e, dur) or (0.0, min(dur, MAX_LEN))

# Transkript-Ausschnitt mit Zeiten relativ zum Video (0 = Anfang von raw.mp4)
with open(latest, encoding="utf-8") as f:
    d = json.load(f)
lines = []
for x in d["transcript"]:
    rs, re_ = x["start"] - SEG_START, x["end"] - SEG_START
    if x["text"] and re_ > 0 and rs < dur:
        lines.append(f"[{max(0, rs):.1f}-{min(dur, re_):.1f}] {x['text']}")
transcript = "\n".join(lines) or "(kein Text)"

# kleine Vorschau fuer Gemini (niedrige Aufloesung, damit die Anfrage klein bleibt)
tmp = tempfile.mkdtemp()
preview = os.path.join(tmp, "preview.mp4")
subprocess.run(
    ["ffmpeg", "-y", "-v", "error", "-i", raw, "-vf", "scale=-2:360", "-r", "12",
     "-c:v", "libx264", "-preset", "veryfast", "-crf", "32",
     "-c:a", "aac", "-b:a", "48k", "-ac", "1", preview],
    check=True,
)

prompt = f"""Du bist ein sehr strenger Editor fuer YouTube Shorts und TikToks aus Twitch-Streams.
Du siehst ein Video von {dur:.0f} Sekunden (Sekunde 0 = Anfang des Videos) mit Ton.
Daraus soll EIN Short werden. Eine erste Auswahl hat diesen Ausschnitt vorgeschlagen:
  Zeitraum: Sekunde {plan_s:.0f} bis {plan_e:.0f}
  Titel: {TITLE}
  Art: {KIND}
Die erste Auswahl hat nur das Transkript gesehen und kann falsch liegen. Entscheide selbst, nachdem
du Bild und Ton wirklich angesehen hast.

Transkript (Zeiten in Sekunden im Video):
{transcript}

Entscheide:
- keep: true nur, wenn im Video ein Moment ist, bei dem ein Fremder beim Scrollen haengen bleibt:
  echte Action im Bild, laute Reaktion, Lachen, Schreck, Chaos, eine klare Pointe. false, wenn der
  Streamer nur erzaehlt oder redet, wenn nichts Auffaelliges passiert, wenn der Moment nur mit
  Vorwissen funktioniert oder wenn das Bild leer oder langweilig ist (Menue, Ladebildschirm, Warten).
  Im Zweifel false.
- score: 1 bis 10, wie sehr bleibt ein Fremder haengen. 8 bis 10 nur fuer Momente, die wirklich
  knallen. Ruhiges Reden hoechstens 4.
- kind: genau eines von action, reaktion, witz, story, gespraech.
- title: deutscher Titel, hoechstens 8 Woerter. Er beschreibt genau das, was du in dem von dir
  gewaehlten Zeitraum siehst und hoerst. Wenn der vorgeschlagene Titel nicht passt, schreib einen
  neuen. Kein Titel, der etwas verspricht, was nicht kommt.
- start, end: Sekunden im Video. Der Clip ist {MIN_LEN:.0f} bis {MAX_LEN:.0f} Sekunden lang. Starte
  kurz vor dem Einstieg (der Zuschauer muss in den ersten Sekunden verstehen, was los ist) und ende
  kurz nach der Pointe, nie mitten im Satz. Nimm so viel wie noetig und so wenig wie moeglich;
  Leerlauf am Anfang oder Ende wird abgeschnitten.
- reason: ein Satz, was im Clip passiert und warum du so entschieden hast.
"""

schema = {
    "type": "OBJECT",
    "properties": {
        "keep": {"type": "BOOLEAN"},
        "score": {"type": "INTEGER"},
        "kind": {"type": "STRING"},
        "title": {"type": "STRING"},
        "start": {"type": "NUMBER"},
        "end": {"type": "NUMBER"},
        "reason": {"type": "STRING"},
    },
    "required": ["keep", "score", "kind", "title", "start", "end", "reason"],
}

checked = True
try:
    # attempts=2: lieber schnell weitermachen als ewig auf Gemini warten
    ans = ask([{"text": prompt}, video_part(preview)], schema, temperature=0.2, attempts=2)
    score = float(ans["score"])
    kind = str(ans.get("kind", KIND)).strip().lower()
    title = str(ans.get("title") or TITLE).strip() or TITLE
    reason = str(ans.get("reason", ""))
    keep = bool(ans.get("keep")) and score >= MIN_SCORE and kind not in TALK_KINDS
    fit = fit_length(float(ans["start"]), float(ans["end"]), dur)
    if fit is None:
        fit = plan  # Gemini-Zeiten unbrauchbar: erste Auswahl behalten
    s, e = fit
except (Exception, SystemExit) as ex:  # SystemExit kommt aus gemini.ask
    key = os.environ.get("GEMINI_API_KEY")
    msg = f"Pruefung fehlgeschlagen: {ex}"
    if key:
        msg = msg.replace(key, "***")
    print(msg[:300])
    checked = False
    keep, score, kind, title, reason = True, 0.0, KIND, TITLE, "Gemini nicht erreichbar, Zeiten und Titel der ersten Auswahl"
    s, e = plan

result = {
    "keep": keep,
    "geprueft": checked,
    "score": score,
    "kind": kind,
    "title": title,
    "title_vorher": TITLE,
    "reason": reason,
    "start_rel": round(s, 2),
    "end_rel": round(e, 2),
    "start_abs": round(SEG_START + s, 2),
    "end_abs": round(SEG_START + e, 2),
    "laenge": round(e - s, 1),
}
with open(out_json, "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=1)
print(json.dumps(result, ensure_ascii=False))

if keep:
    # -ss vor -i bei Neu-Kodierung schneidet bildgenau
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-ss", f"{s:.2f}", "-i", raw, "-t", f"{e - s:.2f}",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
         "-c:a", "aac", "-b:a", "160k", out_mp4],
        check=True,
    )
    print("zugeschnitten:", out_mp4, f"({e - s:.1f} s)")
else:
    print("aussortiert:", title, f"(Score {score:.0f}, Art {kind})")
