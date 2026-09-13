# ApexGuard Historical Backtesting

Backtesting replays a FastF1-derived historical telemetry stream through the **same ApexGuard decision engine** used by `/v1/decide` and aggregates decision, confidence, trap-risk, compliance and latency metrics.

## Build one replay + backtest

```bash
python -m scripts.build_fastf1_replay --year 2025 --event Monaco --session R --driver VER --target-driver NOR --samples 180
python -m scripts.backtest_replay app/data/replays/2025_Monaco_R_VER_NOR_<lap>.json
```

## Build the six-track starter set

```bash
python -m scripts.build_and_backtest_replays --all-default --samples 180
```

The default set is Monaco, Monza, Silverstone, Austria, Spa and Suzuka (2025 race sessions). If a requested driver/lap is unavailable, change the pairing or use a session/lap that exists in FastF1.

## API

- `GET /v1/backtests`
- `GET /v1/backtests/{backtest_id}`
- `POST /v1/backtests/run?replay_id=<id>`

## Interpretation

This is a historical **decision replay**, not a claim that ApexGuard's counterfactual action actually happened. Public FastF1 data does not expose every hidden battery/traffic/rule variable. The replay adapter therefore keeps synthetic fields explicit, and the backtest does not report fake "actual overtakes" or fake ground-truth action accuracy.

## OpenF1 provider

A second provider is now available through `app/replay/openf1_replay.py`.

```bash
python -m scripts.build_openf1_and_backtest --all-default --samples 180
```

OpenF1 historical data is used for observed speed/throttle/brake/DRS, positions,
laps, stints, race-control events and historical overtakes. Battery SOC, detailed
traffic state, circuit geometry proxies and team-specific deployment legality
remain explicitly synthetic/derived. See `OPENF1_REPLAY.md` for the exact
limitations and target-gap handling.
