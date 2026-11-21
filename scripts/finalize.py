"""Baut die Endfassung eines Gewinner-Clips: bestes Whisper-Modell, Untertitel, Layout.

Aufruf: python scripts/finalize.py quelle.mp4 facecam.json ausgabe.mp4

quelle.mp4 ist der geschnittene 16:9-Clip OHNE Layout und Untertitel (trimmed.mp4 aus dem Clip-Job).
Ablauf: Clip mit dem besten Modell transkribieren (Wortzeiten) -> Untertitel (.ass) -> Layout
oben Facecam / unten Gameplay mit eingebrannten Untertiteln.

Einstellungen ueber Umgebungsvariablen (alle optional):
  FINAL_WHISPER_MODEL  Standard "large-v3-turbo". Faellt auf "large-v3" und "small" zurueck,
                       falls das Modell nicht geladen werden kann.
  FINAL_WHISPER_LANG   Sprache, Standard "de". Leer lassen = automatisch erkennen.
  WHISPER_THREADS      Rechenkerne, Standard: alle

Endet mit Fehlercode, wenn etwas schiefgeht. daily_publish.py nimmt dann den Clip ohne Untertitel.
"""
import json
import os
import subprocess
import sys

src, face_json, dst = sys.argv[1:4]
HERE = os.path.dirname(os.path.abspath(__file__))

MODEL = os.environ.get("FINAL_WHISPER_MODEL", "large-v3-turbo")
LANG = os.environ.get("FINAL_WHISPER_LANG", "de") or None
THREADS = int(os.environ.get("WHISPER_THREADS") or (os.cpu_count() or 2))

base = os.path.splitext(dst)[0]
words_json, ass = base + "_words.json", base + ".ass"

dur = float(
    json.loads(
        subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", src],
            check=True, capture_output=True, text=True,
        ).stdout
    )["format"]["duration"]
)

from faster_whisper import WhisperModel  # noqa: E402

model = None
for name in dict.fromkeys([MODEL, "large-v3", "small"]):
    try:
        model = WhisperModel(name, device="cpu", compute_type="int8", cpu_threads=THREADS)
        print("Whisper-Modell:", name)
        break
    except Exception as e:  # noqa: BLE001
        print(f"Modell {name} nicht ladbar: {str(e)[:200]}")
if model is None:
    raise SystemExit("Kein Whisper-Modell ladbar")

segments, info = model.transcribe(
    src,
    language=LANG,
    vad_filter=True,
    word_timestamps=True,
    condition_on_previous_text=False,
    beam_size=5,
)
words = []
for s in segments:
    for w in s.words or []:
        text = w.word.strip()
        if text:
            words.append({"word": text, "start": round(w.start, 2), "end": round(w.end, 2)})
print(f"Transkript: {len(words)} Woerter, Sprache {info.language}")

with open(words_json, "w", encoding="utf-8") as f:
    json.dump({"duration": round(dur, 2), "words": words}, f, ensure_ascii=False)

subprocess.run([sys.executable, os.path.join(HERE, "subtitles.py"), words_json, ass], check=True)
subprocess.run([sys.executable, os.path.join(HERE, "to_split.py"), src, dst, face_json, ass], check=True)
print("fertig:", dst)
