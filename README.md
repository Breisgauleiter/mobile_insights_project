# Mobile Insights Project (MVP Prototype)

Dies ist das Gerüst für ein MVP zur mobilen Version der Insights.gg‑Plattform. Das Projekt besteht aus drei Hauptkomponenten:

1. **Mobile App (React Native oder Flutter – hier als Platzhalter)**
   - Nutzt native Module zur Bildschirmaufnahme via MediaProjection (Android) und ReplayKit (iOS).
   - Ermöglicht Upload von Spielaufnahmen an den Server.

2. **Server (Node.js/Express)**
   - API‑Endpoints zum Hochladen von Videodateien und Abfragen der verarbeiteten Ergebnisse.
   - Ablage der Uploads in einem lokalen `uploads`‑Verzeichnis oder in einem Cloud‑Bucket (Platzhalter).

3. **Machine‑Learning‑Pipeline (Python)**
   - Skript zur ersten Highlight‑Erkennung (Dummy‑Implementation).  
   - Später durch ein YOLO‑/OCR‑basiertes System zu ersetzen.

## Verzeichnisstruktur

```
mobile_insights_project/
  README.md          – Projektübersicht
  server/            – Node.js‑Express‑Server
    index.js         – Einstiegspunkt mit einfachem Upload‑Endpoint
    package.json     – Abhängigkeiten (Express, Multer)
  ml/                – Highlight‑Erkennungs-Pipeline
    highlight_detector.py – Dummy‑Skript für Highlight‑Erkennung
    requirements.txt – Python‑Abhängigkeiten
  mobile_app/        – Platzhalter für die mobile App (Quellcode noch nicht enthalten)
```

## Verwendung (Server)

1. **Node-Abhängigkeiten installieren**
   ```bash
   cd server
   npm install
   ```

2. **Server starten**
   ```bash
   node index.js
   ```
   Der Server akzeptiert POST‑Uploads unter `/upload` und speichert die Datei im `uploads`‑Verzeichnis.

## Verwendung (Machine Learning)

1. **Python‑Abhängigkeiten installieren**
   ```bash
   cd ml
   pip install -r requirements.txt
   ```

2. **Highlight-Erkennung ausführen**
   ```bash
   python highlight_detector.py --video path/to/video.mp4
   ```
   Die aktuelle Implementation ist nur eine Dummy‑Analyse und listet zufällige Zeitstempel als "Highlights" auf. Dieses Skript dient als Platzhalter für zukünftige Modelle (z. B. YOLO‑basiert).

## Hinweis
Die mobile App ist in diesem Repository noch nicht implementiert, da sie native Bibliotheken und ein mobiles Build‑System voraussetzt. Sie soll später als separater Ordner im Unterverzeichnis `mobile_app` entstehen.
