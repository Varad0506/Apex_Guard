"""Build a cached frontend-ready FastF1 replay.

Examples:
    python -m scripts.build_fastf1_replay --year 2025 --event Monaco --session R --driver VER --target-driver NOR
    python -m scripts.build_fastf1_replay --year 2025 --event Monaco --session R --driver VER --target-driver NOR --lap 45 --samples 240
"""
import argparse
import json
from app.replay.fastf1_replay import build_replay


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--year", type=int, required=True)
    p.add_argument("--event", required=True)
    p.add_argument("--session", default="R")
    p.add_argument("--driver", required=True)
    p.add_argument("--target-driver", default=None)
    p.add_argument("--lap", type=int, default=None)
    p.add_argument("--target-lap", type=int, default=None)
    p.add_argument("--samples", type=int, default=180)
    args = p.parse_args()
    payload = build_replay(args.year, args.event, args.session, args.driver,
                           args.target_driver, args.lap, args.target_lap, args.samples)
    print(json.dumps({k: v for k, v in payload.items() if k != "frames"}, indent=2))
    print(f"frames={len(payload['frames'])}")
    print(f"cache_file={payload.get('cache_file')}")


if __name__ == "__main__":
    main()
