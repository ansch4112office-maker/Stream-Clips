"""Transkribiert die Audiospur (mit Wortzeiten fuer Untertitel) und findet Lautstaerke-Spitzen.

Aufruf: python scripts/analyze.py audio.wav ausgabe.json

Die Audiodatei wird in Stuecken (Standard: 15 Minuten) verarbeitet statt in einem Rutsch:
- Der Speicherbedarf bleibt klein, egal wie lang das VOD ist.
- Das Log zeigt nach jedem Stueck den Fortschritt und den Speicherverbrauch. Bricht der Lauf ab,
  sieht man sofort, wo es passiert ist.

Die Ausgabe enthaelt "transcript" (Segmente), "words" (jedes Wort mit Start und Ende in Sekunden)
und "loud_moments".

Einstellungen ueber Umgebungsvariablen (alle optional):
  WHISPER_MODEL    Modellgroesse, Standard "tiny"
  CHUNK_SEC        Laenge eines Stuecks in Sekunden, Standard 900 (wird auf ein Vielfaches von 5 gerundet)
  WHISPER_THREADS  Rechenkerne fuer Whisper, Standard: alle Kerne minus einer (mindestens 1),
                   damit der GitHub-Runner selbst genug Luft hat
"""
import json
import os
import resource
import sys
import time
import wave

import numpy as np
from faster_whisper import WhisperModel

audio_path, out_path = sys.argv[1], sys.argv[2]

MODEL = os.environ.get("WHISPER_MODEL", "tiny")
WIN_SEC = 5  # Fensterbreite fuer die Lautstaerke
CHUNK_SEC = max(WIN_SEC, int(os.environ.get("CHUNK_SEC", "900")) // WIN_SEC * WIN_SEC)
THREADS = int(os.environ.get("WHISPER_THREADS") or max(1, (os.cpu_count() or 2) - 1))
LOUD_FACTOR = 1.6  # ab dem Wievielfachen der ueblichen Lautstaerke ein Fenster als laut gilt
MAX_LOUD = 150  # so viele laute Fenster werden hoechstens gespeichert


def log(*args):
    print(*args, flush=True)


def peak_ram_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # Linux liefert KB


t0 = time.time()

w = wave.open(audio_path, "rb")
rate, channels, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
if rate != 16000 or channels != 1 or width != 2:
    raise SystemExit(
        f"Audio muss 16 kHz, mono, 16 Bit sein, ist aber {rate} Hz, {channels} Kanal, {width * 8} Bit"
    )
total = w.getnframes()
if total == 0:
    raise SystemExit("Die Audiodatei ist leer.")

chunk_frames = CHUNK_SEC * rate
n_chunks = total // chunk_frames + (1 if total % chunk_frames else 0)
log(
    f"Audio: {total / rate / 60:.1f} Minuten, {n_chunks} Stuecke a {CHUNK_SEC // 60} Minuten, "
    f"Modell '{MODEL}', {THREADS} Threads, {os.cpu_count()} Kerne"
)

model = WhisperModel(MODEL, device="cpu", compute_type="int8", cpu_threads=THREADS)

transcript = []  # Segmente mit Zeitstempeln
words = []  # einzelne Woerter mit Zeitstempeln (fuer Untertitel und genaues Schneiden)
rms = []  # Lautstaerke pro 5-Sekunden-Fenster ueber die ganze Datei
language = os.environ.get("WHISPER_LANG") or None  # wird beim ersten sicheren Treffer festgelegt und danach beibehalten

win = rate * WIN_SEC
for idx in range(n_chunks):
    pcm = np.frombuffer(w.readframes(chunk_frames), dtype=np.int16)
    if pcm.size == 0:
        break
    base = idx * CHUNK_SEC  # Startzeit dieses Stuecks in Sekunden

    # 1) Lautstaerke (nur volle Fenster, ein angefangenes Fenster am Ende faellt weg)
    for i in range(0, pcm.size - win + 1, win):
        part = pcm[i : i + win].astype(np.float32)
        rms.append(float(np.sqrt(np.mean(part**2))))

    # 2) Transkript dieses Stuecks, mit Wortzeiten.
    # condition_on_previous_text=False verhindert, dass sich Whisper in Wiederholungen verfaengt.
    samples = pcm.astype(np.float32) / 32768.0
    segments, info = model.transcribe(
        samples,
        vad_filter=True,
        language=language,
        word_timestamps=True,
        condition_on_previous_text=False,
    )
    for s in segments:
        transcript.append(
            {
                "start": round(base + s.start, 1),
                "end": round(base + s.end, 1),
                "text": s.text.strip(),
            }
        )
        for wd in s.words or []:
            text = wd.word.strip()
            if text:
                words.append(
                    {"word": text, "start": round(base + wd.start, 2), "end": round(base + wd.end, 2)}
                )
    # Sprache nur festhalten, wenn Whisper sich sicher ist (sonst im naechsten Stueck neu erkennen)
    if language is None and info.language_probability >= 0.7:
        language = info.language

    del pcm, samples
    done_min = min((idx + 1) * CHUNK_SEC, total / rate) / 60
    log(
        f"[{idx + 1}/{n_chunks}] bis Minute {done_min:.0f} fertig | {len(transcript)} Segmente, "
        f"{len(words)} Woerter | RAM-Spitze {peak_ram_mb():.0f} MB | {time.time() - t0:.0f} s"
    )

w.close()

# 3) laute Momente: Fenster, die deutlich lauter sind als der Median
rms_arr = np.array(rms)
median = (float(np.median(rms_arr)) if rms_arr.size else 0.0) or 1.0

loud = [
    {"start": i * WIN_SEC, "end": i * WIN_SEC + WIN_SEC, "factor": round(v / median, 2)}
    for i, v in enumerate(rms)
    if v / median >= LOUD_FACTOR
]
loud.sort(key=lambda x: x["factor"], reverse=True)

with open(out_path, "w", encoding="utf-8") as f:
    json.dump(
        {
            "language": language or "unbekannt",
            "duration_sec": round(total / rate),
            "loud_moments": loud[:MAX_LOUD],
            "transcript": transcript,
            "words": words,
        },
        f,
        ensure_ascii=False,
        separators=(",", ":"),
    )
log("fertig:", len(transcript), "Segmente,", len(words), "Woerter,", len(loud), "laute Momente")
