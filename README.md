# Mobile Insights

MVP-Prototyp zur automatischen Highlight-Erkennung in Gameplay-Videos. Videos werden per Web-Oberfläche hochgeladen und von einer Multi-Signal-ML-Pipeline analysiert, die Frame-Differenz, Audio-Ereignisse (Whisper) und Farb-Bursts kombiniert, um Highlights automatisch zu klassifizieren.

## Funktionsumfang

- **Multi-Signal-Highlight-Erkennung** — drei unabhängige Detektoren, deren Ergebnisse zusammengeführt und nach Score gewichtet werden:
  - *Frame-Differenz* – erkennt starke Bewegungsmomente
  - *Whisper-Audio* – erkennt MLBB-Ansager-Events (Kills, Objectives, Multi-Kills)
  - *Farb-Burst* – erkennt plötzliche Farb-Aufflackerer (z. B. Effekte, Fähigkeiten)
- **Web-UI** — Drag-and-Drop-Upload, Echtzeit-Fortschrittsanzeige, Video-Player mit Highlight-Navigation
- **Fortschrittsverfolgung** — der Server streamt den ML-Fortschritt live an den Browser (0–100 % pro Stage, ETA-Anzeige)
- **Docker-Deployment** — einzelnes Image für Server und ML-Pipeline; Produktionsstack hinter Traefik mit HTTPS
- **CI/CD** — GitHub Actions für Server (Node.js 18/20) und ML-Pipeline (Python 3.11/3.12)

## Architektur

```
Browser (Web-UI)
     │  POST /upload  GET /uploads  GET /results/:id  GET /video/:filename
     ▼
Node.js/Express Server  (server/index.js)
     │  spawnt subprocess
     ▼
Python ML-Pipeline  (ml/highlight_detector.py)
  ├── Frame-Differenz-Detektor  (opencv)
  ├── Audio-Detektor            (ffmpeg + OpenAI Whisper)
  └── Farb-Burst-Detektor       (opencv)
```

Im Produktionsbetrieb läuft der Container hinter einem **Traefik**-Reverse-Proxy mit automatischem Let's Encrypt-Zertifikat.

## Verzeichnisstruktur

```
mobile_insights_project/
  README.md                  – Projektübersicht
  Dockerfile                 – Kombiniertes Image (Server + ML)
  docker-compose.yml         – Lokaler Standardstack
  docker-compose.local.yml   – Lokale Overrides (reduzierte ML-Last)
  docker-compose.prod.yml    – Produktionsstack mit Traefik-Labels
  .env.example               – Vorlage für Umgebungsvariablen
  deploy/
    deploy.sh                – Deployment-Skript (rsync + SSH)
  server/
    index.js                 – Express-Server, API-Endpoints
    public/index.html        – Web-UI (Single Page)
    package.json             – Abhängigkeiten (Express, Multer)
    __tests__/               – Jest-Tests
  ml/
    highlight_detector.py    – Multi-Signal-Pipeline (Einstiegspunkt)
    audio_detector.py        – Whisper-Audio-Detektor
    color_detector.py        – Farb-Burst-Detektor
    requirements.txt         – Python-Abhängigkeiten
    tests/                   – pytest-Tests
  mobile_app/                – Platzhalter für die mobile App
```

## Schnellstart (Docker)

Voraussetzung: Docker Desktop oder Docker Engine mit Compose-Plugin.

```bash
# Vollständigen Stack starten (Server + ML-Pipeline)
docker compose up --build

# Mit reduzierten ML-Einstellungen für schwächere Rechner
docker compose -f docker-compose.yml -f docker-compose.local.yml up --build
```

Die Web-Oberfläche ist anschließend unter **http://localhost:3000** erreichbar.

```bash
# Stack stoppen
docker compose down
```

### Lokale Entwicklung ohne Docker

```bash
# Node-Abhängigkeiten installieren und Server starten
cd server
npm install
node index.js

# Python-Abhängigkeiten installieren
cd ml
pip install -r requirements.txt

# ML-Pipeline manuell aufrufen
python highlight_detector.py --video path/to/video.mp4 --format json
```

## Web-UI

Die Web-Oberfläche (`server/public/index.html`) bietet:

- **Drag-and-Drop-Upload** — Videos per Drag & Drop oder Datei-Dialog hochladen (MP4, MOV, AVI, WebM; max. 500 MB)
- **Upload-Fortschritt** — Fortschrittsbalken während des Uploads
- **ML-Fortschritt** — Echtzeit-Anzeige der aktuellen Analyse-Stage (Frame-Diff → Audio → Farbe) mit Prozentzahl und ETA
- **Highlight-Karten** — jedes Highlight zeigt Zeitstempel, Event-Typ (z. B. `savage`, `triple_kill`, `action`), Score-Balken und verwendete Signalquellen
- **Video-Player** — integrierter Player mit Klick-zu-Seek: Klick auf eine Highlight-Karte springt direkt zur entsprechenden Stelle im Video

## API-Übersicht

| Methode | Pfad | Beschreibung |
|---------|------|-------------|
| `POST` | `/upload` | Video hochladen (multipart, Feld `video`) |
| `GET` | `/uploads` | Liste aller Uploads mit Status und Fortschritt |
| `GET` | `/results/:id` | Highlight-Ergebnisse für ein Video abrufen |
| `GET` | `/video/:filename` | Video streamen (Range-Support für Seeking) |
| `GET` | `/health` | Health-Check |

### Antwortformat `/results/:id`

```json
{
  "id": "1234567890-video.mp4",
  "status": "done",
  "highlights": [
    {
      "timestamp": 42.5,
      "score": 95.0,
      "type": "savage",
      "sources": ["audio", "frame_diff"]
    }
  ]
}
```

Mögliche Statuswerte: `pending` | `processing` | `done` | `error`

## ML-Pipeline konfigurieren

Die Pipeline wird über Umgebungsvariablen gesteuert (siehe `.env.example`):

| Variable | Standard | Beschreibung |
|----------|---------|-------------|
| `WHISPER_MODEL` | `tiny` | Whisper-Modellgröße (`tiny`, `small`, `medium`, `large`) |
| `ML_SKIP_FRAMES` | `0` | Jeden N-ten Frame analysieren (0 = alle) |
| `ML_MAX_WIDTH` | `0` | Frames auf diese Breite skalieren (0 = original) |
| `ML_NO_AUDIO` | `0` | Audio-Erkennung deaktivieren (`1` = aus) |
| `ML_NO_COLOR` | `0` | Farb-Burst-Erkennung deaktivieren (`1` = aus) |

Für lokale Entwicklung mit Docker Desktop empfiehlt sich `docker-compose.local.yml` (setzt `ML_SKIP_FRAMES=3` und `ML_MAX_WIDTH=480`, um RAM-Verbrauch zu begrenzen).

## Produktions-Deployment

Der Produktionsstack (`docker-compose.prod.yml`) läuft auf einem Linux-Server mit Traefik als Reverse-Proxy.

### Voraussetzungen auf dem Server

- Docker mit aktivem `traefik-public`-Netzwerk
- Traefik mit `websecure`-Entrypoint und `main-resolver` für Let's Encrypt
- SSH-Zugang zum Produktionsserver

### Deployment ausführen

```bash
# Einmalig: .env anlegen und Domain setzen
cp .env.example .env
# DOMAIN=meine-domain.example.com in .env eintragen

# Deployment starten (rsync + Docker Build auf dem Server)
bash deploy/deploy.sh deploy
```

Weitere Befehle:

```bash
bash deploy/deploy.sh status    # Container-Status anzeigen
bash deploy/deploy.sh logs      # Live-Logs ausgeben
bash deploy/deploy.sh restart   # Container neu starten
bash deploy/deploy.sh stop      # Stack stoppen
bash deploy/deploy.sh cleanup   # Alte Docker-Images löschen
```

## CI/CD

Das Repository enthält zwei GitHub-Actions-Workflows:

| Workflow | Trigger | Aufgaben |
|---------|---------|---------|
| **Server CI** (`.github/workflows/server-ci.yml`) | Push/PR auf `main`, Pfad `server/**` | ESLint, Jest-Tests (Node.js 18 & 20) |
| **ML Pipeline CI** (`.github/workflows/ml-ci.yml`) | Push/PR auf `main`, Pfad `ml/**` | ruff-Linting, pytest (Python 3.11 & 3.12) |

## Hinweis zur mobilen App

Die mobile App (`mobile_app/`) ist derzeit noch ein Platzhalter. Sie soll als React-Native-App native Bildschirmaufnahme (MediaProjection/ReplayKit) und den Upload an den Server ermöglichen.
