Stream-Clips
Regeln
Nur Streamer, die in streamers.txt stehen (Clips offiziell erlaubt). Steht der Streamer nicht drin: nicht weitermachen, Nutzer darauf hinweisen.
Streamername und Original-Link immer in die Beschreibung jedes Clips schreiben.
Ablauf
Nutzer schickt VOD-Link und Streamername.
Nutzer startet in GitHub unter Actions den Workflow "Transkribieren" mit Link und Streamer. Ergebnis landet in data/latest.json.
data/latest.json lesen. Enthalten sind loud_moments (Lautstaerke-Spitzen mit Faktor ueber dem Median) und transcript (Text mit Zeitstempeln).
5 bis 10 Clips auswaehlen, jeweils 20 bis 45 Sekunden. Gute Kandidaten:
laute Reaktion plus passender Text im Transkript (Lachen, Schreien, "was?!")
klare Pointe oder Wendung in wenigen Saetzen
Moment, der ohne Vorwissen verstaendlich ist
Start 3 bis 5 Sekunden vor dem Hoehepunkt, Ende kurz nach der Pointe
Ausgabe pro Clip als Tabelle oder Liste:
Start- und Endzeit (HH:MM:SS)
Hook fuer die ersten Sekunden
Titel mit #Shorts
3 bis 5 Hashtags
Beschreibung mit Streamername und Original-Link
eine Zeile Begruendung, warum der Moment funktionieren koennte
Nutzer startet pro Clip den Workflow "Clip schneiden" mit Link, Start, Ende und Name. Die fertige 9:16-Datei liegt danach als Artifact beim Workflow-Lauf.
Grenzen
Nur Text und Ton-Daten, keine Bilder. Rein visuelle Momente werden nicht erkannt.
"Viral" ist eine Einschaetzung nach Mustern, keine Garantie.
Bei Songs im Hintergrund auf Musikrechte hinweisen.
