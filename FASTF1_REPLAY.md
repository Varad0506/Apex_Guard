# FastF1 Replay

ApexGuard can now turn a real FastF1 session/lap into a frontend-ready replay stream.

## Install

```bash
pip install -r requirements.txt
```

FastF1 is optional for the core decision API, but is required for real-session replay.

## Build a replay

```bash
python -m scripts.build_fastf1_replay --year 2025 --event Monaco --session R --driver VER --target-driver NOR --samples 180
```

If `--lap` is omitted, ApexGuard uses the selected driver's fastest lap. If `--target-driver` is omitted, it selects another available driver from the session.

Cached JSON is written under:

`app/data/replays/`

## API

Start the server:

```bash
uvicorn app.main:app --reload
```

List cached replays:

`GET /v1/replays`

Load one:

`GET /v1/replays/{replay_id}`

Generate directly from FastF1:

`GET /v1/replays/fastf1?year=2025&event=Monaco&session=R&driver=VER&target_driver=NOR&samples=180`

## Data honesty

FastF1 supplies driver telemetry such as speed, distance, throttle/brake and DRS. ApexGuard currently synthesizes fields that cannot be reconstructed reliably from the selected telemetry alone: battery SOC, tyre grip, traffic counts, deployment budget, minimum reserve and overtaking difficulty. The replay payload explicitly labels those fields in `synthetic_fields` so the UI can distinguish measured telemetry from ApexGuard demo assumptions.
