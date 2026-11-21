"""Findet Chat-Spitzen und Zuschauer-Clips und traegt sie als "chat_moments" in latest.json ein.

Aufruf: python scripts/chat_peaks.py data/latest.json chat.json [twitch_clips.json]

chat.json:         Chat-Replay von TwitchDownloaderCLI (fetch_chat.py). Fehlt die Datei, zaehlen nur Zuschauer-Clips.
twitch_clips.json: Liste {"start", "end", "views"} in VOD-Sekunden (fetch_chat.py, optional).

Idee: Ein 10-Sekunden-Fenster (alle 5 Sekunden neu) zaehlt als Chat-Spitze, wenn
  - die Nachrichtenrate mindestens CHAT_MIN_RATE (Standard 3 pro Sekunde) erreicht UND
  - mindestens CHAT_MIN_RATIO (Standard 3) mal so hoch ist wie der Median der Fenster der letzten
    CHAT_BASE_MIN Minuten (Standard 30). Der Median ist robust: Eine lange Hype-Phase zieht die
    Baseline nicht mit nach oben, wie es ein Durchschnitt tun wuerde.
Der Chat reagiert erst mit 5 bis 15 Sekunden Verzoegerung. Deshalb steht neben der Chat-Zeit
(start/end) auch event_start/event_end: der Zeitraum, in dem der Anlass vermutlich passiert ist.

Staerke (factor) = Anstieg * (1 + Anteil Hype-Nachrichten + 0,5 * Anteil identischer Nachrichten)
+ bis zu 3 Punkte fuer Zuschauer-Clips an der Stelle. Zuschauer-Clips ohne Chat-Spitze zaehlen als
eigener Moment (type "clip").
"""
import json
import os
import re
import sys
from collections import Counter

import numpy as np

latest = sys.argv[1]
chat_path = sys.argv[2] if len(sys.argv) > 2 else ""
clips_path = sys.argv[3] if len(sys.argv) > 3 else ""

STEP, WIN = 5, 10  # Sekunden: ein 10-Sekunden-Fenster, alle 5 Sekunden neu
BASE_SEC = int(float(os.environ.get("CHAT_BASE_MIN", "30")) * 60)
MIN_RATE = float(os.environ.get("CHAT_MIN_RATE", "3"))
MIN_RATIO = float(os.environ.get("CHAT_MIN_RATIO", "3"))
MIN_BASE = 0.5  # Baseline mindestens so hoch, damit im toten Chat nicht schon 0,2 -> 1 als krass gilt
MIN_HISTORY = 12  # Fenster Vorlauf, bevor eine Baseline gilt (1 Minute)
LAG_MAX, LAG_MIN = 15.0, 5.0  # so viele Sekunden vorher ist der Anlass vermutlich passiert
GROUP_GAP = 15.0
CLIP_MAX_EVENT = 45.0
MAX_MOMENTS = 150

HYPE = re.compile(
    r"\b(rip|xd|wtf|clip|lol|omg)\b|lul|kekw|omegalul|lmao|lmfao|haha|pog|monka|pepega|icant|\?|!!",
    re.I,
)


def is_hype(body):
    if HYPE.search(body):
        return True
    letters = [c for c in body if c.isalpha()]
    return len(letters) >= 4 and sum(c.isupper() for c in letters) / len(letters) >= 0.7


def norm(body):
    return re.sub(r"\s+", " ", body.lower()).strip()


def load_chat(path):
    if not path or not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    items = []
    for c in data.get("comments", []):
        try:
            t = float(c["content_offset_seconds"])
        except (KeyError, TypeError, ValueError):
            continue
        body = str((c.get("message") or {}).get("body") or "").strip()
        if body:
            items.append((t, body))
    items.sort(key=lambda x: x[0])
    return items


def load_clips(path):
    if not path or not os.path.exists(path):
        return []
    try:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        return sorted(
            ({"start": float(c["start"]), "end": float(c["end"]), "views": int(c.get("views", 0))} for c in raw),
            key=lambda c: c["start"],
        )
    except (OSError, ValueError, KeyError, TypeError):
        return []


def cluster_clips(clips):
    """Fasst sich ueberlappende Zuschauer-Clips zusammen."""
    out = []
    for c in clips:
        if out and c["start"] < out[-1]["end"]:
            o = out[-1]
            o["end"] = max(o["end"], c["end"])
            o["count"] += 1
            o["views"] += c["views"]
        else:
            out.append({"start": c["start"], "end": c["end"], "count": 1, "views": c["views"]})
    return out


def overlap(a0, a1, b0, b1):
    return a0 < b1 and a1 > b0


with open(latest, encoding="utf-8") as f:
    d = json.load(f)
lo = float(d.get("offset_sec", 0))
hi = lo + float(d["duration_sec"])

msgs = load_chat(chat_path)
if msgs and lo > 60 and msgs[-1][0] < lo:
    print(f"Chat-Zeiten scheinen relativ zum Ausschnitt zu sein (letzte Nachricht bei {msgs[-1][0]:.0f} s), verschiebe um {lo:.0f} s")
    msgs = [(t + lo, b) for t, b in msgs]
if msgs:
    print(f"Chat: {len(msgs)} Nachrichten von {msgs[0][0]:.0f} s bis {msgs[-1][0]:.0f} s (VOD-Zeit), Ausschnitt {lo:.0f} bis {hi:.0f} s")

moments = []
info = {"messages": len(msgs)}

n_win = int((hi - lo - WIN) // STEP) + 1
if msgs and n_win >= MIN_HISTORY + 3:
    times = np.array([m[0] for m in msgs])
    hype = np.array([1 if is_hype(m[1]) else 0 for m in msgs])
    hype_cum = np.concatenate(([0], np.cumsum(hype)))

    starts = lo + STEP * np.arange(n_win)
    i0 = np.searchsorted(times, starts, side="left")
    i1 = np.searchsorted(times, starts + WIN, side="left")
    counts = i1 - i0
    rate = counts / WIN

    hist = max(MIN_HISTORY, BASE_SEC // STEP)
    base = np.full(n_win, np.nan)
    for i in range(MIN_HISTORY + 2, n_win):
        # Fenster bis i-2 (i-1 ueberlappt noch mit Fenster i)
        base[i] = np.median(rate[max(0, i - 1 - hist) : i - 1])
    ratio = np.where(np.isnan(base), 0.0, rate / np.maximum(np.nan_to_num(base), MIN_BASE))
    info["median_rate"] = round(float(np.median(rate)), 2)

    idx = np.where((rate >= MIN_RATE) & (ratio >= MIN_RATIO))[0]
    groups, cur = [], []
    for i in idx:
        if cur and (i - cur[-1]) * STEP > GROUP_GAP:
            groups.append(cur)
            cur = []
        cur.append(int(i))
    if cur:
        groups.append(cur)

    for g in groups:
        p = max(g, key=lambda i: ratio[i])
        n = int(counts[p])
        hype_share = float(hype_cum[i1[p]] - hype_cum[i0[p]]) / max(1, n)
        dup_share = 0.0
        if n >= 6:
            top = Counter(norm(b) for _, b in msgs[i0[p] : i1[p]]).most_common(1)[0][1]
            dup_share = top / n
        factor = float(ratio[p]) * (1 + hype_share + 0.5 * dup_share)
        s, e = float(starts[g[0]]), float(starts[g[-1]] + WIN)
        ev_s = max(lo, s - LAG_MAX)
        ev_e = min(hi, max(ev_s + 5.0, e - LAG_MIN))
        moments.append(
            {
                "type": "chat",
                "start": round(s, 1),
                "end": round(e, 1),
                "event_start": round(ev_s, 1),
                "event_end": round(ev_e, 1),
                "rate": round(float(rate[p]), 1),
                "base": round(float(base[p]), 2),
                "ratio": round(float(ratio[p]), 1),
                "emote_share": round(hype_share, 2),
                "dup_share": round(dup_share, 2),
                "clips": 0,
                "factor": round(factor, 2),
            }
        )
elif msgs:
    print("Ausschnitt zu kurz fuer eine Baseline, keine Chat-Spitzen.")

# Zuschauer-Clips
clusters = [c for c in cluster_clips(load_clips(clips_path)) if c["end"] > lo and c["start"] < hi]
used = set()
for m in moments:
    for k, c in enumerate(clusters):
        if overlap(c["start"], c["end"], m["event_start"] - 5, m["event_end"] + 5):
            m["clips"] += c["count"]
            used.add(k)
    if m["clips"]:
        m["factor"] = round(m["factor"] + min(3, m["clips"]), 2)
for k, c in enumerate(clusters):
    if k in used:
        continue
    s, e = max(lo, c["start"]), min(hi, c["start"] + CLIP_MAX_EVENT, c["end"])
    moments.append(
        {
            "type": "clip",
            "start": round(s, 1),
            "end": round(e, 1),
            "event_start": round(s, 1),
            "event_end": round(max(e, s + 5.0), 1),
            "clips": c["count"],
            "views": c["views"],
            "factor": round(min(8.0, 3.0 + c["count"]), 2),
        }
    )

moments.sort(key=lambda m: m["factor"], reverse=True)
moments = moments[:MAX_MOMENTS]
d["chat_moments"] = moments
d["chat_info"] = info
with open(latest, "w", encoding="utf-8") as f:
    json.dump(d, f, ensure_ascii=False, separators=(",", ":"))

print(f"Chat-Spitzen: {sum(m['type'] == 'chat' for m in moments)}, Zuschauer-Clip-Momente: {sum(m['type'] == 'clip' for m in moments)}")
for m in moments[:10]:
    extra = f"{m['rate']}/s statt {m['base']}/s, Hype {m['emote_share']:.0%}" if m["type"] == "chat" else f"{m['clips']} Clips"
    print(f"  x{m['factor']:<5} {m['type']:<5} Anlass {int(m['event_start'])}-{int(m['event_end'])} s  ({extra}, Clips {m['clips']})")
