"""Kleiner Helfer fuer die Gemini-API (nur Standardbibliothek, nichts zu installieren).

Der Schluessel kommt aus der Umgebungsvariable GEMINI_API_KEY (GitHub-Secret).
Optional: GEMINI_MODEL setzen, dann wird dieses Modell zuerst probiert.
"""
import base64
import json
import os
import time
import urllib.error
import urllib.request

API = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

# Wird der Reihe nach probiert, falls ein Name nicht (mehr) existiert oder das Limit erreicht ist.
# Aktuelle Namen stehen in Google AI Studio unter "Modelle".
FALLBACK_MODELS = [
    "gemini-3.8-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
]


def image_part(path):
    """Ein JPG als Bild-Teil fuer die Anfrage."""
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    return {"inline_data": {"mime_type": "image/jpeg", "data": data}}


def ask(parts, schema, temperature=0.3, attempts=4):
    """Schickt parts (Text und Bilder) an Gemini und gibt die Antwort als JSON-Objekt zurueck.

    attempts = Versuche pro Modell (bei Limit oder Ueberlastung wird kurz gewartet)."""
    key = os.environ["GEMINI_API_KEY"]

    models = []
    if os.environ.get("GEMINI_MODEL"):
        models.append(os.environ["GEMINI_MODEL"])
    models += [m for m in FALLBACK_MODELS if m not in models]

    body = json.dumps(
        {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {
                "temperature": temperature,
                "responseMimeType": "application/json",
                "responseSchema": schema,
            },
        }
    ).encode()

    last_error = "unbekannt"
    for model in models:
        for attempt in range(attempts):
            req = urllib.request.Request(
                API.format(model=model),
                data=body,
                headers={"Content-Type": "application/json", "x-goog-api-key": key},
            )
            try:
                with urllib.request.urlopen(req, timeout=180) as r:
                    resp = json.load(r)
                text = resp["candidates"][0]["content"]["parts"][0]["text"]
                result = json.loads(text)
                print("Gemini-Modell:", model)
                return result
            except urllib.error.HTTPError as e:
                detail = e.read().decode(errors="replace")[:300]
                last_error = f"{model}: HTTP {e.code} {detail}"
                if e.code in (429, 500, 503):
                    # Limit erreicht oder kurz ueberlastet: warten und nochmal
                    time.sleep(10 * (attempt + 1))
                    continue
                break  # z. B. 404 (Modell gibt es nicht): naechstes Modell
            except (KeyError, IndexError, json.JSONDecodeError) as e:
                last_error = f"{model}: unerwartete Antwort ({e!r})"
                break
            except (urllib.error.URLError, TimeoutError) as e:
                last_error = f"{model}: Netzwerkfehler ({e!r})"
                time.sleep(5)
                continue
        print("Modell uebersprungen:", last_error)

    raise SystemExit("Gemini-Aufruf fehlgeschlagen: " + last_error)
