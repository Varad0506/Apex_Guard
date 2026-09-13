from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.replay.fastf1_replay import build_replay
from app.evaluation.backtest import backtest_replay

# A compact, diverse starter set. Driver pairings can be overridden from CLI.
DEFAULT_REPLAYS = [
    (2025, "Monaco", "R", "VER", "NOR"),
    (2025, "Monza", "R", "VER", "LEC"),
    (2025, "Silverstone", "R", "NOR", "VER"),
    (2025, "Austria", "R", "VER", "NOR"),
    (2025, "Spa", "R", "VER", "LEC"),
    (2025, "Suzuka", "R", "VER", "NOR"),
]


def main():
    p = argparse.ArgumentParser(description="Build FastF1 replays and run ApexGuard backtests")
    p.add_argument("--year", type=int, default=None)
    p.add_argument("--event", default=None)
    p.add_argument("--session", default="R")
    p.add_argument("--driver", default="VER")
    p.add_argument("--target-driver", default=None)
    p.add_argument("--samples", type=int, default=180)
    p.add_argument("--all-default", action="store_true")
    args = p.parse_args()

    if args.all_default:
        jobs = DEFAULT_REPLAYS
    else:
        if not args.event or not args.year:
            p.error("provide --year and --event, or use --all-default")
        jobs = [(args.year, args.event, args.session, args.driver, args.target_driver)]

    out_dir = Path(__file__).resolve().parents[1] / "app" / "data" / "evaluations" / "backtests"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for year, event, session, driver, target in jobs:
        print(f"Building {year} {event} {session}: {driver} vs {target}")
        replay = build_replay(year, event, session, driver, target, samples=args.samples, cache=True)
        key = f"{year}_{event.replace(' ', '_')}_{session}_{driver}_{target}"
        replay_path = Path(__file__).resolve().parents[1] / "app" / "data" / "replays" / f"{key}.json"
        replay_path.write_text(json.dumps(replay, indent=2), encoding="utf-8")
        result = backtest_replay(replay)
        result_path = out_dir / f"{key}_backtest.json"
        result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        summary.append({"id": key, "replay": str(replay_path), "backtest": str(result_path), "metrics": result["metrics"]})
    summary_path = out_dir / "backtest_summary.json"
    summary_path.write_text(json.dumps({"runs": summary}, indent=2), encoding="utf-8")
    print(json.dumps({"summary": str(summary_path), "runs": len(summary)}, indent=2))


if __name__ == "__main__":
    main()
