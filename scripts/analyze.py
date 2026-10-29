"""Transkribiert die Audiospur und findet Lautstaerke-Spitzen.

Aufruf: python scripts/analyze.py audio.wav ausgabe.json
"""
import json
import sys
import wave

import numpy as np
from faster_whisper import WhisperModel

audio_path, out_path = sys.argv[1], sys.argv[2]

# 1) Transkript mit Zeitstempeln
model = WhisperModel("small", device="cpu", compute_type="int8")
segments, info = model.transcribe(audio_path, vad_filter=True)
transcript = [
    {"start": round(s.start, 1), "end": round(s.end, 1), "text": s.text.strip()}
    for s in segments
]

# 2) Lautstaerke pro 5-Sekunden-Fenster
with wave.open(audio_path, "rb") as w:
    rate = w.getframerate()
    data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32)

win = rate * 5
rms = [
    float(np.sqrt(np.mean(data[i : i + win] ** 2)))
    for i in range(0, len(data) - win, win)
]
rms_arr = np.array(rms)
median = float(np.median(rms_arr)) or 1.0

loud = [
    {"start": i * 5, "end": i * 5 + 5, "factor": round(v / median, 2)}
    for i, v in enumerate(rms)
    if v / median >= 2.0
]
loud.sort(key=lambda x: x["factor"], reverse=True)

with open(out_path, "w", encoding="utf-8") as f:
    json.dump(
        {
            "language": info.language,
            "duration_sec": round(len(data) / rate),
            "loud_moments": loud[:40],
            "transcript": transcript,
        },
        f,
        ensure_ascii=False,
        indent=1,
    )
print("fertig:", len(transcript), "Segmente,", len(loud), "laute Momente")
