from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.replay.openf1_replay import build_replay
from app.evaluation.backtest import backtest_replay

DEFAULT_REPLAYS = [
    (2025, "Monaco", "Race", "VER", "NOR"),
    (2025, "Monza", "Race", "VER", "LEC"),
    (2025, "Silverstone", "Race", "NOR", "VER"),
    (2025, "Austria", "Race", "VER", "NOR"),
    (2025, "Spa", "Race", "VER", "LEC"),
    (2025, "Suzuka", "Race", "VER", "NOR"),
]


def main():
    p = argparse.ArgumentParser(description="Build OpenF1 historical replays and run ApexGuard backtests")
    p.add_argument("--year", type=int)
    p.add_argument("--event")
    p.add_argument("--session", default="Race")
    p.add_argument("--driver", default="VER")
    p.add_argument("--target-driver", default=None)
    p.add_argument("--samples", type=int, default=180)
    p.add_argument("--all-default", action="store_true")
    p.add_argument("--no-cache", action="store_true")
    args = p.parse_args()

    if args.all_default:
        jobs = DEFAULT_REPLAYS
    else:
        if not args.year or not args.event:
            p.error("provide --year and --event, or use --all-default")
        jobs = [(args.year, args.event, args.session, args.driver, args.target_driver)]

    root = Path(__file__).resolve().parents[1]
    replay_dir = root / "app" / "data" / "replays"
    out_dir = root / "app" / "data" / "evaluations" / "backtests"
    replay_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    for year, event, session, driver, target in jobs:
        print(f"Building OpenF1 {year} {event} {session}: {driver} vs {target}")
        replay = build_replay(year, event, session, driver, target, samples=args.samples, cache=not args.no_cache)
        target = replay["target_driver"]
        key = f"{year}_{event.replace(' ', '_')}_{session}_{driver}_{target}_openf1"
        replay_path = replay_dir / f"{key}.json"
        replay_path.write_text(json.dumps(replay, indent=2), encoding="utf-8")
        result = backtest_replay(replay)
        result_path = out_dir / f"{key}_backtest.json"
        result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        summary.append({"id": key, "replay": str(replay_path), "backtest": str(result_path), "metrics": result["metrics"]})

    summary_path = out_dir / "openf1_backtest_summary.json"
    summary_path.write_text(json.dumps({"provider": "OpenF1", "runs": summary}, indent=2), encoding="utf-8")
    print(json.dumps({"summary": str(summary_path), "runs": len(summary)}, indent=2))


if __name__ == "__main__":
    main()
