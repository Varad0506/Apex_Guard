"""Train the optional LSTM + learned-HMM racer-pattern layer.

Usage:
  python -m scripts.train_rival_pattern_model --replays app/data/replays
  python -m scripts.train_rival_pattern_model --replays app/data/replays/openf1_cache

The labels are weak tactical labels derived from the existing observable
belief model.  This is intentional: OpenF1 does not expose proprietary ERS
intent. The LSTM learns temporal patterns across historical replay windows
rather than claiming direct access to hidden driver strategy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from app.models import rival_pattern_model
from app.models.rival_pattern_model import (
    FEATURE_DIM, WINDOW_SIZE, TACTICAL_STATES, RacerPatternLSTM, make_feature_vector
)

try:
    import torch
    from torch.utils.data import DataLoader, TensorDataset
except Exception as exc:
    raise SystemExit("PyTorch is required for offline LSTM training. Install the optional PPO dependencies.") from exc


def _iter_replay_files(root: Path):
    if root.is_file() and root.suffix.lower() == ".json":
        yield root
        return
    for p in sorted(root.rglob("*.json")):
        if "artifacts" in p.parts:
            continue
        yield p


def _frame_feature(frame: dict[str, Any]) -> list[float]:
    ego = frame.get("ego", {})
    target = frame.get("target", {})
    track = frame.get("track", {})
    return make_feature_vector(
        gap_s=target.get("gap_s", 1.0),
        relative_speed_kph=target.get("relative_speed_kph", 0.0),
        sector_kind=track.get("segment_type", "CORNER"),
        dt_s=0.5,
        throttle=ego.get("throttle_pct", 0.0),
        brake=ego.get("brake_pct", 0.0),
        speed_kph=ego.get("speed_kph", 0.0),
        drs=1.0 if track.get("drs_available") else 0.0,
        soc_pct=ego.get("soc_pct", 50.0),
        tyre_grip=ego.get("tyre_grip_estimate", 0.82),
        position_delta=0.0,
    )


def _proxy_label(frame: dict[str, Any]) -> int:
    """Observable proxy target; never uses hidden ERS fields."""
    target = frame.get("target", {})
    track = frame.get("track", {})
    gap = float(target.get("gap_s", 1.0))
    rel = float(target.get("relative_speed_kph", 0.0))
    delta = float(target.get("recent_sector_delta_s", 0.0))
    close = max(0.0, min(1.0, (1.5 - gap) / 1.5))
    slowing = max(0.0, min(1.0, -rel / 20.0))
    attacking = max(0.0, min(1.0, rel / 20.0))
    degrading = max(0.0, min(1.0, -delta * 2.0))
    scores = {
        "HARVESTING": 0.9 * slowing + 0.2 * (1.0 - close),
        "DEFENDING": 0.8 * close + 0.2 * degrading,
        "ATTACKING": 0.9 * attacking + 0.4 * close + 0.1 * bool(track.get("drs_available")),
        "CONSERVING": 0.7 * degrading + 0.2 * (1.0 - close),
    }
    return max(range(len(TACTICAL_STATES)), key=lambda i: scores[TACTICAL_STATES[i]])


def load_sequences(root: Path):
    X, y = [], []
    files = list(_iter_replay_files(root))
    for path in files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        frames = payload.get("frames") if isinstance(payload, dict) else payload
        if not isinstance(frames, list) or len(frames) < WINDOW_SIZE + 1:
            continue
        features = [_frame_feature(f) for f in frames]
        labels = [_proxy_label(f) for f in frames]
        for end in range(WINDOW_SIZE - 1, len(frames) - 1):
            X.append(features[end - WINDOW_SIZE + 1:end + 1])
            y.append(labels[end + 1])
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int64), files


def learn_transition(labels: np.ndarray):
    n = len(TACTICAL_STATES)
    counts = np.ones((n, n), dtype=np.float64) * 0.5
    for a, b in zip(labels[:-1], labels[1:]):
        counts[int(a), int(b)] += 1.0
    probs = counts / counts.sum(axis=1, keepdims=True)
    return probs.tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replays", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    X, y, files = load_sequences(args.replays)
    if len(X) < 8:
        raise SystemExit(
            f"Need at least 8 historical windows; found {len(X)}. "
            "Build/download OpenF1 replays first."
        )

    model = RacerPatternLSTM()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(X), torch.from_numpy(y)),
        batch_size=args.batch_size,
        shuffle=True,
    )

    model.train()
    for _ in range(args.epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()

    artifact_dir = Path(__file__).resolve().parents[1] / "app" / "data" / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), artifact_dir / "racer_pattern_lstm.pt")

    # Learn HMM transitions from the same chronological weak labels.
    # The runtime layer treats this matrix as advisory and still blends in
    # current observation evidence.
    transition = learn_transition(y)
    (artifact_dir / "racer_pattern_hmm.json").write_text(
        json.dumps({
            "model": "racer-pattern-hmm-v1",
            "states": list(TACTICAL_STATES),
            "transition": transition,
            "training_files": [str(p) for p in files],
            "window_size": WINDOW_SIZE,
            "label_type": "observable_proxy",
        }, indent=2),
        encoding="utf-8",
    )

    print(json.dumps({
        "status": "trained",
        "windows": int(len(X)),
        "replay_files": len(files),
        "window_size": WINDOW_SIZE,
        "features": FEATURE_DIM,
        "artifacts": [
            str(artifact_dir / "racer_pattern_lstm.pt"),
            str(artifact_dir / "racer_pattern_hmm.json"),
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
