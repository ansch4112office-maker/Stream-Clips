"""Laedt den Chat-Replay eines VODs und (optional) die Zuschauer-Clips dazu.

Aufruf: python scripts/fetch_chat.py VOD_URL START END chat.json twitch_clips.json

START/END im Format HH:MM:SS, END darf "inf" sein (= bis zum Ende).

Chat: TwitchDownloaderCLI (wird bei Bedarf von GitHub geladen, braucht "gh" und GH_TOKEN).
Zuschauer-Clips: Twitch-API (kostenlos), nur wenn TWITCH_CLIENT_ID und TWITCH_CLIENT_SECRET gesetzt
sind. Der Streamer kommt aus der Umgebungsvariable STREAMER.

Dieses Skript bricht nie ab. Was nicht klappt, wird im Log gemeldet, die Datei fehlt dann einfach.
"""
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.parse
import urllib.request
import zipfile
from datetime import datetime, timedelta, timezone

vod_url, start_s, end_s, chat_out, clips_out = sys.argv[1:6]


def to_sec(t):
    t = (t or "").strip().lower()
    if not t or t == "inf":
        return None
    h, m, s = (int(x) for x in t.split(":"))
    return h * 3600 + m * 60 + s


m = re.search(r"videos/(\d+)", vod_url)
if not m:
    print("Keine VOD-ID im Link gefunden, kein Chat.")
    sys.exit(0)
vod_id = m.group(1)
start, end = to_sec(start_s) or 0, to_sec(end_s)


def get_cli():
    found = shutil.which("TwitchDownloaderCLI")
    if found:
        return found
    shutil.rmtree("tdl", ignore_errors=True)
    os.makedirs("tdl")
    subprocess.run(
        ["gh", "release", "download", "-R", "lay295/TwitchDownloader", "-p", "*Linux-x64*", "-D", "tdl"],
        check=True,
    )
    for z in glob.glob("tdl/*.zip"):
        with zipfile.ZipFile(z) as zf:
            zf.extractall("tdl")
    for p in glob.glob("tdl/**/TwitchDownloaderCLI", recursive=True):
        os.chmod(p, 0o755)
        return os.path.abspath(p)
    raise RuntimeError("TwitchDownloaderCLI nicht im Download gefunden")


def fetch_chat():
    cli = get_cli()
    cmd = [cli, "chatdownload", "--id", vod_id, "-o", chat_out, "-b", str(start)]
    if end is not None:
        cmd += ["-e", str(end)]
    print("Chat laden:", " ".join(cmd[1:]))
    subprocess.run(cmd, check=True, timeout=40 * 60)
    print("Chat geladen:", os.path.getsize(chat_out) // 1024, "KB")


def api(url, token, cid):
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}", "Client-Id": cid})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def parse_dur(s):
    mm = re.fullmatch(r"(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?", s or "")
    h, mi, se = (int(x or 0) for x in mm.groups()) if mm else (0, 0, 0)
    return h * 3600 + mi * 60 + se


def fetch_viewer_clips():
    cid, secret = os.environ.get("TWITCH_CLIENT_ID"), os.environ.get("TWITCH_CLIENT_SECRET")
    streamer = os.environ.get("STREAMER", "").strip().lower()
    if not (cid and secret and streamer):
        print("Keine Twitch-API-Zugangsdaten, ueberspringe Zuschauer-Clips.")
        return
    body = urllib.parse.urlencode(
        {"client_id": cid, "client_secret": secret, "grant_type": "client_credentials"}
    ).encode()
    with urllib.request.urlopen("https://id.twitch.tv/oauth2/token", data=body, timeout=30) as r:
        token = json.load(r)["access_token"]

    vid = api(f"https://api.twitch.tv/helix/videos?id={vod_id}", token, cid)["data"][0]
    created = datetime.fromisoformat(vid["created_at"].replace("Z", "+00:00"))
    length = parse_dur(vid.get("duration", ""))
    user = api(f"https://api.twitch.tv/helix/users?login={streamer}", token, cid)["data"][0]["id"]

    a = (created - timedelta(minutes=5)).astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    b = (created + timedelta(seconds=length, hours=2)).astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    clips, cursor = [], ""
    for _ in range(10):  # hoechstens 1000 Clips
        url = f"https://api.twitch.tv/helix/clips?broadcaster_id={user}&first=100&started_at={a}&ended_at={b}"
        if cursor:
            url += "&after=" + cursor
        res = api(url, token, cid)
        for c in res.get("data", []):
            off = c.get("vod_offset")
            if str(c.get("video_id")) != vod_id or off is None:
                continue
            clips.append(
                {"start": float(off), "end": float(off) + float(c.get("duration", 30)), "views": int(c.get("view_count", 0))}
            )
        cursor = (res.get("pagination") or {}).get("cursor", "")
        if not cursor:
            break
    lo, hi = start, end if end is not None else 10**9
    clips = [c for c in clips if c["end"] > lo and c["start"] < hi]
    with open(clips_out, "w", encoding="utf-8") as f:
        json.dump(clips, f)
    print(f"Zuschauer-Clips: {len(clips)} im Ausschnitt")


for name, fn in (("Chat", fetch_chat), ("Zuschauer-Clips", fetch_viewer_clips)):
    try:
        fn()
    except Exception as e:  # noqa: BLE001 - darf nie abbrechen
        print(f"{name} nicht geladen: {str(e)[:300]}")
sys.exit(0)
