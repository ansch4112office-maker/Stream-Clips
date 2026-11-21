"""Sucht pro Streamer das neueste fertige, noch nicht verarbeitete VOD und startet dafuer auto_clips.yml.

Aufruf: python scripts/daily_start.py

Schreibt runs.json (welcher Lauf gehoert zu welchem Streamer) und data/processed_vods.txt
(VODs, die schon dran waren, damit nichts doppelt geclippt wird).

Optional (Umgebungsvariablen):
  MAX_HOURS   so viele Stunden vom VOD-Anfang werden hoechstens analysiert, Standard 10
  MIN_MIN     kuerzere VODs werden uebersprungen, Standard 30 Minuten
  MAX_AGE_H   aeltere VODs werden uebersprungen, Standard 36 Stunden
"""
import json
import os
import subprocess
import time
from datetime import datetime, timezone

MAX_HOURS = float(os.environ.get("MAX_HOURS", "10"))
MIN_SEC = float(os.environ.get("MIN_MIN", "30")) * 60
MAX_AGE = float(os.environ.get("MAX_AGE_H", "36")) * 3600
PROCESSED = "data/processed_vods.txt"


def sh(cmd):
    return subprocess.run(cmd, capture_output=True, text=True)


def hms(sec):
    sec = int(sec)
    return f"{sec // 3600:02d}:{sec % 3600 // 60:02d}:{sec % 60:02d}"


with open("streamers.txt", encoding="utf-8") as f:
    streamers = [l.strip().lower() for l in f if l.strip() and not l.startswith("#")]

os.makedirs("data", exist_ok=True)
done = set()
if os.path.exists(PROCESSED):
    with open(PROCESSED, encoding="utf-8") as f:
        done = set(f.read().split())

began = datetime.now(timezone.utc)
now = time.time()
started = []

for st in streamers:
    url = f"https://www.twitch.tv/{st}/videos?filter=archives&sort=time"
    r = sh(["yt-dlp", "-j", "--playlist-items", "1-3", url])
    vods = []
    for ln in r.stdout.splitlines():
        try:
            vods.append(json.loads(ln))
        except json.JSONDecodeError:
            pass
    if not vods:
        print(f"{st}: keine VODs gefunden ({r.stderr.strip()[:200]})")
        continue

    pick = None
    for v in vods:  # neueste zuerst
        vid = str(v.get("id"))
        dur, ts = v.get("duration"), v.get("timestamp")
        if not dur or not ts:
            continue
        if vid in done:
            continue
        if ts + dur > now - 900:  # Stream laeuft noch oder ist gerade erst zu Ende
            print(f"{st}: VOD {vid} noch nicht fertig")
            continue
        if now - (ts + dur) > MAX_AGE:
            continue
        if dur < MIN_SEC:
            continue
        pick = (vid, v.get("webpage_url") or f"https://www.twitch.tv/videos/{vid}", dur, v.get("title", ""))
        break

    if not pick:
        print(f"{st}: kein neues VOD")
        continue

    vid, vod_url, dur, vod_title = pick
    end = hms(min(dur, MAX_HOURS * 3600))
    print(f"{st}: starte {vod_url} (0 bis {end}, VOD ist {hms(dur)} lang)")
    g = sh(["gh", "workflow", "run", "auto_clips.yml",
            "-f", f"vod_url={vod_url}", "-f", f"streamer={st}",
            "-f", "start=00:00:00", "-f", f"end={end}"])
    if g.returncode != 0:
        print(f"{st}: Start fehlgeschlagen: {g.stderr.strip()[:300]}")
        continue
    done.add(vid)
    started.append({"streamer": st, "vod": vod_url, "vod_title": vod_title})

with open(PROCESSED, "w", encoding="utf-8") as f:
    f.write("\n".join(sorted(done)) + "\n")

runs = []
if started:
    time.sleep(45)  # den neuen Laeufen Zeit geben, aufzutauchen
    out = sh(["gh", "run", "list", "--workflow", "auto_clips.yml", "--event", "workflow_dispatch",
              "--limit", "50", "--json", "databaseId,displayTitle,createdAt"]).stdout
    listed = json.loads(out or "[]")
    for s in started:
        mine = [
            x for x in listed
            if x["displayTitle"] == f"Auto-Clips {s['streamer']}"
            and datetime.fromisoformat(x["createdAt"].replace("Z", "+00:00")) >= began
        ]
        if mine:
            s["run_id"] = max(mine, key=lambda x: x["createdAt"])["databaseId"]
            runs.append(s)
        else:
            print(f"{s['streamer']}: Lauf nicht gefunden")

with open("runs.json", "w", encoding="utf-8") as f:
    json.dump(runs, f, ensure_ascii=False, indent=1)
print(f"{len(runs)} Laeufe gestartet")
