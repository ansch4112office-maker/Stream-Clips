"""Wandelt ein 16:9-Video in 9:16 um: oben die Facecam, unten das Gameplay (leicht gezoomt,
ohne Facecam im Bild). Optional werden Untertitel eingebrannt.

Aufruf: python scripts/to_split.py eingabe.mp4 ausgabe.mp4 facecam.json [untertitel.ass]

Ohne erkannte Facecam wird das alte Layout (to_vertical.py) benutzt, ebenfalls mit Untertiteln.
"""
import json
import os
import subprocess
import sys

src, dst, face_json = sys.argv[1], sys.argv[2], sys.argv[3]
ass = sys.argv[4] if len(sys.argv) > 4 and os.path.exists(sys.argv[4]) else None
HERE = os.path.dirname(os.path.abspath(__file__))

W, H = 1080, 1920
TOP_MIN, TOP_MAX = 600, 900  # Hoehe des Facecam-Bereichs
MIN_ZOOM, MAX_ZOOM = 1.1, 2.2  # Gameplay mindestens leicht gezoomt
MARGIN = 0.02  # Sicherheitsabstand zur Facecam (Anteil der Bildbreite/-hoehe)


def even(n):
    return int(n) // 2 * 2


def fallback(reason):
    print("Altes Layout wegen:", reason)
    cmd = [sys.executable, os.path.join(HERE, "to_vertical.py"), src, dst]
    if ass:
        cmd.append(ass)
    subprocess.run(cmd, check=True)
    sys.exit(0)


def probe(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "json", path],
        check=True, capture_output=True, text=True,
    ).stdout
    s = json.loads(out)["streams"][0]
    return int(s["width"]), int(s["height"])


def overlaps(a, b):
    return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]


def game_crop(sw, sh, face, aspect):
    """Sucht einen Ausschnitt (x, y, w, h) mit passendem Seitenverhaeltnis, der die Facecam
    nicht beruehrt und moeglichst mittig liegt. Zoomt bei Bedarf weiter hinein."""
    # groesster Ausschnitt mit dem Seitenverhaeltnis, der ins Bild passt
    if sw / sh >= aspect:
        base_h, base_w = sh, sh * aspect
    else:
        base_w, base_h = sw, sw / aspect

    blocked = (
        face[0] - MARGIN * sw, face[1] - MARGIN * sh,
        face[2] + MARGIN * sw, face[3] + MARGIN * sh,
    )
    zoom = MIN_ZOOM
    while zoom <= MAX_ZOOM + 1e-9:
        cw, ch = base_w / zoom, base_h / zoom
        best = None
        for i in range(13):
            for j in range(7):
                x = (sw - cw) * i / 12
                y = (sh - ch) * j / 6
                if overlaps((x, y, x + cw, y + ch), blocked):
                    continue
                dist = abs(x + cw / 2 - sw / 2) / sw + abs(y + ch / 2 - sh / 2) / sh
                if best is None or dist < best[0]:
                    best = (dist, x, y)
        if best:
            return even(best[1]), even(best[2]), even(cw), even(ch)
        zoom += 0.05
    return None


try:
    with open(face_json, encoding="utf-8") as f:
        face = json.load(f)
except (OSError, json.JSONDecodeError):
    fallback("facecam.json fehlt oder ist kaputt")

if not face.get("found"):
    fallback("keine Facecam erkannt (" + str(face.get("reason", "ohne Angabe")) + ")")

sw, sh = probe(src)
fx0, fy0 = even(face["box"][0] * sw), even(face["box"][1] * sh)
fx1, fy1 = even(face["box"][2] * sw), even(face["box"][3] * sh)
fw, fh = fx1 - fx0, fy1 - fy0
if fw < 64 or fh < 64:
    fallback("Facecam-Ausschnitt zu klein")

top_h = even(min(TOP_MAX, max(TOP_MIN, round(W * fh / fw))))
bottom_h = H - top_h

crop = game_crop(sw, sh, (fx0, fy0, fx1, fy1), W / bottom_h)
if crop is None:
    fallback("kein Gameplay-Ausschnitt ohne Facecam gefunden")
cx, cy, cw, ch = crop
print(f"Facecam {fw}x{fh} bei {fx0},{fy0}; Gameplay {cw}x{ch} bei {cx},{cy}; oben {top_h}px, unten {bottom_h}px")

sub_filter = f",ass={ass}" if ass else ""
filter_complex = (
    "[0:v]split=2[a][b];"
    f"[a]crop={fw}:{fh}:{fx0}:{fy0},"
    f"scale={W}:{top_h}:force_original_aspect_ratio=increase:flags=lanczos,"
    f"crop={W}:{top_h},setsar=1[top];"
    f"[b]crop={cw}:{ch}:{cx}:{cy},scale={W}:{bottom_h}:flags=lanczos,setsar=1[bot];"
    f"[top][bot]vstack=inputs=2{sub_filter},format=yuv420p[v]"
)

subprocess.run(
    [
        "ffmpeg", "-y", "-v", "error", "-i", src,
        "-filter_complex", filter_complex,
        "-map", "[v]", "-map", "0:a?",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-c:a", "aac", "-b:a", "160k",
        "-movflags", "+faststart",
        dst,
    ],
    check=True,
)
print("fertig:", dst)
