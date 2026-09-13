# OpenF1 Historical Replay + Backtesting

ApexGuard can use **OpenF1 as a second historical replay provider** alongside FastF1. OpenF1 exposes public historical F1 telemetry and timing data from 2023 onward, including car data, intervals, positions, laps, stints, race-control events and overtakes.

The adapter converts those observations into the same `DecisionRequest`-shaped frames consumed by the ApexGuard backtester. This means the decision engine remains unchanged.

## Quick start

Install the existing requirements, then run one replay:

```bash
python -m scripts.build_openf1_and_backtest --year 2025 --event Monaco --session Race --driver VER --target-driver NOR --samples 180
```

Run the six-track starter set:

```bash
python -m scripts.build_openf1_and_backtest --all-default --samples 180
```

The six defaults are Monaco, Monza, Silverstone, Austria, Spa and Suzuka (2025 race sessions).

## API

Generate an OpenF1 replay directly:

```text
GET /v1/replays/openf1?year=2025&event=Monaco&session=Race&driver=VER&target_driver=NOR&samples=180
```

Existing FastF1 routes remain available.

## What is observed vs synthetic

Observed/derived from OpenF1:

- speed, throttle, brake and DRS
- driver positions over time
- interval to the car immediately ahead
- lap/stint information
- race-control flags/events
- historical overtake events
- relative speed between selected drivers

Synthetic/latent in the ApexGuard replay:

- battery SOC / ERS reserve
- tyre grip estimate
- detailed traffic density
- deployment budget
- team-specific legal state
- circuit distance remaining / braking-zone geometry
- overtake difficulty
- opponent hidden tactical/energy state

The backtest therefore evaluates **decision behavior on historical observations**, not a fabricated counterfactual claim that ApexGuard would have caused or prevented a historical overtake.

## Important target-gap rule

OpenF1's `interval` is the gap to the car immediately ahead, not an arbitrary selected target. The adapter uses it as the target gap only when OpenF1 position data confirms the selected target is ahead. Otherwise it uses a conservative synthetic fallback and records that source in the frame.

## Caching

OpenF1 responses are cached under:

```text
app/data/replays/openf1_cache/
```

This is intentional: a six-race backtest can require multiple historical endpoint calls, and caching avoids repeatedly downloading the same data.

## Research/demo positioning

OpenF1 is an unofficial community API and is not affiliated with Formula 1 or the FIA. Use it as a historical research/replay provider, not as an official timing or regulatory source.
