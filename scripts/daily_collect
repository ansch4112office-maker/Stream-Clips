"""Wartet, bis die gestarteten auto_clips-Laeufe fertig sind, und laedt ihre Clips herunter.

Aufruf: python scripts/daily_collect.py

Liest runs.json (von daily_start.py) und legt die Clips in clips/<streamer>/ ab.

Optional:
  MAX_WAIT_MIN  so lange wird hoechstens gewartet, Standard 330 Minuten
"""
import json
import os
import subprocess
import time

MAX_WAIT = float(os.environ.get("MAX_WAIT_MIN", "330")) * 60


def sh(cmd):
    return subprocess.run(cmd, capture_output=True, text=True)


with open("runs.json", encoding="utf-8") as f:
    runs = json.load(f)

pending = {r["run_id"]: r for r in runs}
deadline = time.time() + MAX_WAIT
while pending and time.time() < deadline:
    for rid in list(pending):
        out = sh(["gh", "run", "view", str(rid), "--json", "status"]).stdout
        try:
            status = json.loads(out)["status"]
        except (json.JSONDecodeError, KeyError):
            continue
        if status == "completed":
            print(f"fertig: {pending[rid]['streamer']} (Lauf {rid})")
            del pending[rid]
    if pending:
        time.sleep(300)

for r in pending.values():
    print(f"nicht rechtzeitig fertig: {r['streamer']} (Lauf {r['run_id']})")

os.makedirs("clips", exist_ok=True)
for r in runs:
    if r["run_id"] in pending:
        continue
    d = os.path.join("clips", r["streamer"])
    os.makedirs(d, exist_ok=True)
    g = sh(["gh", "run", "download", str(r["run_id"]), "-n", "fertige-clips", "-D", d])
    if g.returncode != 0:
        print(f"{r['streamer']}: keine Clips ({g.stderr.strip()[:150]})")
