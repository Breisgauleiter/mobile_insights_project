# Copilot Instructions – Mobile Insights Project

## Projektübersicht
Mobile Insights ist ein MVP-Prototyp zur mobilen Version der Insights.gg-Plattform.
Das Ziel: Spielaufnahmen mobil aufnehmen, an einen Server senden und per ML-Pipeline automatisch Highlights erkennen.

## Architektur (Monorepo)

| Komponente | Pfad | Stack |
|---|---|---|
| Backend API | `server/` | Node.js, Express, Multer |
| ML Pipeline | `ml/` | Python 3.11+, OpenCV, NumPy (später YOLO/OCR) |
| Mobile App | `mobile_app/` | React Native (noch Platzhalter) |

## Coding Conventions

### Allgemein
- Sprache im Code, Commits und PRs: **Englisch**
- Kommunikation mit dem User: **Deutsch**
- Branchbenennungen: `feat/`, `fix/`, `chore/`, `docs/` + kurze Beschreibung (kebab-case)
- Commits: Conventional Commits (`feat:`, `fix:`, `chore:`, `docs:`, `test:`, `ci:`)
- PRs werden via **Merge Commit** gemergt (kein Squash, kein Rebase)

### Node.js (server/)
- ES-Module-Syntax wird noch nicht genutzt (CommonJS mit `require`)
- Express als Framework
- Multer für File-Uploads
- Tests mit Jest (noch einzurichten)
- Linting mit ESLint

### Python (ml/)
- Python 3.11+
- Type Hints verwenden
- Docstrings für öffentliche Funktionen
- Tests mit pytest
- Linting mit ruff
- Dependencies in `requirements.txt`

### Mobile App (mobile_app/)
- React Native (Einrichtung steht noch aus)
- TypeScript bevorzugt

## Workflow
- Issues beschreiben Aufgaben – Agents arbeiten diese ab
- Jede Änderung kommt über einen Feature-Branch + PR
- CI muss grün sein bevor gemergt wird
- PR-Zyklus: Agent PR → Copilot Review → Fixes → Merge

## Wichtige Pfade
- Server-Einstiegspunkt: `server/index.js`
- ML-Einstiegspunkt: `ml/highlight_detector.py`
- CI/CD: `.github/workflows/`
- Issue Templates: `.github/ISSUE_TEMPLATE/`
