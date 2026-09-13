# ApexGuard Frontend

React + Vite cockpit UI for the ApexGuard backend, built from `APEXGUARD_UI_UX_SPEC.md`.
Every number on screen is traced to a real field returned by the FastAPI backend — nothing
in this app is mocked or invented.

## Run it

1. Start the backend (from the backend project root):
   ```bash
   uvicorn app.main:app --reload --port 8000
   ```
2. Install and run the frontend:
   ```bash
   npm install
   npm run dev
   ```
   Opens on `http://localhost:5173`. The dev server proxies `/v1/*` to `http://localhost:8000`
   (see `vite.config.js`), so no CORS setup is needed locally.

   To point at a different backend host, set `VITE_API_BASE` in a `.env` file, e.g.
   `VITE_API_BASE=http://localhost:8000`.

## What's implemented

All 12 screens from the design spec, each wired to real endpoints:

| Screen | Backend calls |
|---|---|
| Overview | (session-derived from decisions run elsewhere) |
| Decision Engine | `POST /v1/decide` |
| Race Replay | `GET /v1/replays`, `GET /v1/replays/{id}`, `POST /v1/decide` |
| Simulate (quantum-inspired) | `POST /v1/simulate/quantum-inspired` |
| Opponent Belief | `POST /v1/decide` (reads `opponent_belief`, ticked via `battle_id`) |
| Traffic | `POST /v1/decide` (reads `traffic_summary`) |
| Scenario Library | `GET /v1/scenarios`, `GET /v1/scenarios/{id}`, `POST /v1/decide` |
| Track Library | static track metadata (backend has no `/v1/tracks` endpoint; values are read from the 10 track JSON files shipped in `app/data/tracks`) |
| Performance Lab | `GET /v1/backtests/openf1_backtest_summary` |
| Backtests | `GET /v1/backtests`, `GET /v1/backtests/{id}`, `POST /v1/backtests/run` |
| Audit Trail | client-tracked session history of decisions (the backend only exposes `GET /v1/audits/{id}` — a single lookup, not a list) |
| System Health | `GET /v1/health`, `GET /v1/health/ready` |

## Design system

Tokens (color, type, layout) live in `src/styles.css` and follow the UI/UX spec: warm-graphite
base, four functional signal colors (never decorative), monospace for live-updating numbers,
one orchestrated pipeline-lighting motion per decision, no SaaS card-kit shadows/radii.
