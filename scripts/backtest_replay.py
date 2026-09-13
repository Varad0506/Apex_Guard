from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.evaluation.backtest import backtest_file


def main():
    parser = argparse.ArgumentParser(description="Run ApexGuard historical replay backtest")
    parser.add_argument("replay", help="Path to FastF1 replay JSON")
    parser.add_argument("--output", help="Output JSON path")
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()
    output = args.output or str(Path(args.replay).with_name(Path(args.replay).stem + "_backtest.json"))
    result = backtest_file(args.replay, output, args.max_frames)
    print(json.dumps({"output": output, "metrics": result["metrics"], "action_counts": result["action_counts"]}, indent=2))


if __name__ == "__main__":
    main()
