"""Wandelt ein 16:9-Video in 9:16 um (Original mittig, unscharfer Hintergrund).
 
Aufruf: python scripts/to_vertical.py eingabe.mp4 ausgabe.mp4
"""
import subprocess
import sys
 
src, dst = sys.argv[1], sys.argv[2]
 
filter_complex = (
    "[0:v]scale=1080:1920:force_original_aspect_ratio=increase,"
    "crop=1080:1920,boxblur=30:5[bg];"
    "[0:v]scale=1080:-2[fg];"
    "[bg][fg]overlay=(W-w)/2:(H-h)/2"
)
 
subprocess.run(
    [
        "ffmpeg", "-y", "-i", src,
        "-filter_complex", filter_complex,
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-c:a", "aac", "-b:a", "160k",
        dst,
    ],
    check=True,
)
print("fertig:", dst)
 
