"""Waehlt aus allen Clips des Tages die besten aus und laedt sie auf YouTube hoch.

Aufruf: python scripts/daily_publish.py

Erwartet die Clips in clips/<streamer>/NAME.mp4 mit NAME.txt (Titel) und NAME.json (review.json).
Gemini schaut einmal ueber alle Kandidaten und waehlt die besten, das Skript achtet darauf, dass
nicht alles vom selben Streamer kommt. Wenn Gemini ausfaellt, entscheidet der Score aus der Pruefung.

Secrets als Umgebungsvariablen: YT_CLIENT_ID, YT_CLIENT_SECRET, YT_REFRESH_TOKEN, GEMINI_API_KEY

Optional:
  MAX_UPLOADS         hoechstens so viele Videos pro Tag, Standard 4
  MAX_PER_STREAMER    hoechstens so viele davon pro Streamer, Standard 2
  PRIVACY             public (Standard), unlisted oder private
"""
import glob
import json
import os
import sys

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload

MAX_UPLOADS = int(os.environ.get("MAX_UPLOADS", "4"))
MAX_PER = int(os.environ.get("MAX_PER_STREAMER", "2"))
PRIVACY = os.environ.get("PRIVACY", "public")

cands = []
for mp4 in sorted(glob.glob("clips/**/*.mp4", recursive=True)):
    base = mp4[:-4]
    streamer = mp4.split(os.sep)[1]
    meta = {}
    if os.path.exists(base + ".json"):
        with open(base + ".json", encoding="utf-8") as f:
            meta = json.load(f)
    title = meta.get("title", "")
    if not title and os.path.exists(base + ".txt"):
        with open(base + ".txt", encoding="utf-8") as f:
            title = f.read().strip()
    cands.append({
        "id": len(cands), "file": mp4, "streamer": streamer, "title": title or "Twitch Clip",
        "kind": meta.get("kind", ""), "score": float(meta.get("score", 0)),
        "reason": meta.get("reason", ""), "laenge": meta.get("laenge", 0),
    })

if not cands:
    print("Keine Clips heute, nichts hochzuladen.")
    sys.exit(0)
print(f"{len(cands)} Kandidaten")


def limit(ids):
    """Wendet MAX_UPLOADS und MAX_PER_STREAMER auf eine Rangliste (Liste von ids) an."""
    out, count = [], {}
    for i in ids:
        c = cands[i]
        if count.get(c["streamer"], 0) >= MAX_PER:
            continue
        count[c["streamer"]] = count.get(c["streamer"], 0) + 1
        out.append(i)
        if len(out) == MAX_UPLOADS:
            break
    return out


by_score = [c["id"] for c in sorted(cands, key=lambda c: c["score"], reverse=True)]
ranked = by_score
try:
    from gemini import ask

    listing = "\n".join(
        f"{c['id']}: [{c['streamer']}] \"{c['title']}\" ({c['kind']}, {c['laenge']} s, Score {c['score']:.0f}) {c['reason']}"
        for c in cands
    )
    prompt = f"""Du bist Chefredakteur eines YouTube-Shorts-Kanals mit Twitch-Highlights.
Unten stehen die Clips von heute. Waehle die {MAX_UPLOADS} besten, bei denen ein Fremder beim
Scrollen haengen bleibt (Action, laute Reaktion, klare Pointe). Achte auf Abwechslung: nicht alles vom
selben Streamer, nicht dreimal dieselbe Art von Moment. Gib die ids in der Reihenfolge von
bestem zu schwaechstem Clip zurueck, hoechstens {MAX_UPLOADS}.

{listing}
"""
    schema = {
        "type": "OBJECT",
        "properties": {"picks": {"type": "ARRAY", "items": {"type": "INTEGER"}}},
        "required": ["picks"],
    }
    ans = ask([{"text": prompt}], schema, temperature=0.2, attempts=2)
    picks = [int(i) for i in ans.get("picks", []) if 0 <= int(i) < len(cands)]
    if picks:
        ranked = list(dict.fromkeys(picks)) + [i for i in by_score if i not in picks]
except (Exception, SystemExit) as e:
    print("Gemini-Auswahl fehlgeschlagen, nehme den Score:", str(e)[:200])

chosen = limit(ranked)
print("Auswahl:", [(cands[i]["streamer"], cands[i]["title"]) for i in chosen])

creds = Credentials(
    None,
    refresh_token=os.environ["YT_REFRESH_TOKEN"],
    token_uri="https://oauth2.googleapis.com/token",
    client_id=os.environ["YT_CLIENT_ID"],
    client_secret=os.environ["YT_CLIENT_SECRET"],
    scopes=["https://www.googleapis.com/auth/youtube.upload"],
)
yt = build("youtube", "v3", credentials=creds)

summary = ["## Hochgeladen", ""]
for i in chosen:
    c = cands[i]
    st = c["streamer"]
    tag = "".join(ch for ch in st if ch.isalnum())
    body = {
        "snippet": {
            "title": c["title"][:100],
            "description": f"{c['title']}\n\n#{tag} #Clips #Twitch #viral #Shorts\n\nQuelle: https://www.twitch.tv/{st}",
            "tags": [tag, "Clips", "Twitch", "viral", "Shorts"],
            "categoryId": "20",
            "defaultLanguage": "de",
        },
        "status": {"privacyStatus": PRIVACY, "selfDeclaredMadeForKids": False},
    }
    try:
        req = yt.videos().insert(
            part="snippet,status", body=body,
            media_body=MediaFileUpload(c["file"], mimetype="video/mp4", resumable=True),
        )
        resp = None
        while resp is None:
            _, resp = req.next_chunk()
        link = f"https://youtu.be/{resp['id']}"
        print("hochgeladen:", c["title"], link)
        summary.append(f"- [{c['title']}]({link}) ({st})")
    except HttpError as e:
        msg = str(e)
        print("Upload fehlgeschlagen:", c["title"], msg[:300])
        summary.append(f"- FEHLER bei \"{c['title']}\": {msg[:150]}")
        if "quotaExceeded" in msg or "uploadLimitExceeded" in msg:
            break

path = os.environ.get("GITHUB_STEP_SUMMARY")
if path:
    with open(path, "a", encoding="utf-8") as f:
        f.write("\n".join(summary) + "\n")
