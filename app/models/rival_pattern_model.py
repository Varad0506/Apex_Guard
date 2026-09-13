"""Learned temporal racer-pattern layer.

The module learns *temporal* tactical patterns from previous-race telemetry.
An optional LSTM encodes a recent telemetry window; a learned HMM transition
matrix provides persistence between tactical states.  Both are advisory:
if the model artifact is absent or PyTorch is unavailable, callers fall back
to the existing deterministic HMM/proxy belief model.

No live training occurs in the API process.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import json
import math

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except Exception:
    torch = None
    nn = None
    TORCH_AVAILABLE = False


TACTICAL_STATES = ("HARVESTING", "DEFENDING", "ATTACKING", "CONSERVING")
FEATURE_DIM = 11
WINDOW_SIZE = 15
HIDDEN_DIM = 32

_ACTIVE_MODEL_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "artifacts" / "racer_pattern_lstm.pt"
)
_ACTIVE_HMM_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "artifacts" / "racer_pattern_hmm.json"
)


def _clamp(v: float) -> float:
    return max(-1.0, min(1.0, float(v)))


def _norm(v: float, scale: float) -> float:
    return _clamp(float(v) / float(scale or 1.0))


def make_feature_vector(
    gap_s: float,
    relative_speed_kph: float,
    sector_kind: str,
    dt_s: float,
    throttle: float = 0.0,
    brake: float = 0.0,
    speed_kph: float = 0.0,
    drs: float = 0.0,
    soc_pct: float = 50.0,
    tyre_grip: float = 0.85,
    position_delta: float = 0.0,
) -> List[float]:
    """Convert observable race state into a bounded LSTM feature vector."""
    sector = str(sector_kind or "CORNER").upper()
    straight = 1.0 if sector in {"STRAIGHT", "DRS_STRAIGHT"} else -1.0
    drs_signal = 1.0 if drs else -1.0
    return [
        _norm(gap_s - 1.0, 2.0),
        _norm(relative_speed_kph, 50.0),
        straight,
        _norm(dt_s - 0.5, 1.0),
        _norm(throttle, 100.0),
        _norm(brake, 100.0),
        _norm(speed_kph, 360.0),
        drs_signal,
        _norm(soc_pct - 50.0, 50.0),
        _clamp(tyre_grip * 2.0 - 1.0),
        _norm(position_delta, 5.0),
    ]


@dataclass
class RivalHistoryWindow:
    values: List[List[float]]

    def __init__(self, values: Optional[Iterable[Iterable[float]]] = None):
        self.values = [list(v) for v in (values or [])][-WINDOW_SIZE:]

    def push(self, **kwargs) -> None:
        if kwargs and "gap_s" in kwargs:
            vector = make_feature_vector(**kwargs)
        else:
            vector = list(kwargs)
        if len(vector) != FEATURE_DIM:
            raise ValueError(f"expected {FEATURE_DIM} features, got {len(vector)}")
        self.values.append(vector)
        self.values = self.values[-WINDOW_SIZE:]

    def is_ready(self) -> bool:
        return len(self.values) >= WINDOW_SIZE

    def as_list(self) -> List[List[float]]:
        return list(self.values)


@dataclass
class LearnedHMM:
    transition: List[List[float]]

    def __post_init__(self):
        n = len(TACTICAL_STATES)
        if len(self.transition) != n or any(len(row) != n for row in self.transition):
            raise ValueError("HMM transition matrix must be 4x4")
        self.transition = [_normalize(row) for row in self.transition]

    def predict(self, previous: Dict[str, float]) -> Dict[str, float]:
        prior = _normalize([previous.get(s, 0.0) for s in TACTICAL_STATES])
        out = []
        for j in range(len(TACTICAL_STATES)):
            out.append(sum(prior[i] * self.transition[i][j] for i in range(len(TACTICAL_STATES))))
        total = sum(out) or 1.0
        return {s: out[i] / total for i, s in enumerate(TACTICAL_STATES)}


def _normalize(values):
    if isinstance(values, dict):
        total = sum(max(0.0, float(v)) for v in values.values()) or 1.0
        return {k: max(0.0, float(v)) / total for k, v in values.items()}
    vals = [max(0.0, float(v)) for v in values]
    total = sum(vals) or 1.0
    return [v / total for v in vals]


def proxy_label(belief: Dict[str, float]) -> int:
    """Return the argmax tactical-state index used for weakly-labelled training."""
    return max(range(len(TACTICAL_STATES)), key=lambda i: belief.get(TACTICAL_STATES[i], 0.0))


if TORCH_AVAILABLE:
    class RacerPatternLSTM(nn.Module):
        def __init__(self, input_dim: int = FEATURE_DIM, hidden_dim: int = HIDDEN_DIM,
                     num_classes: int = len(TACTICAL_STATES)):
            super().__init__()
            self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
            self.head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                      nn.Linear(hidden_dim, num_classes))

        def forward(self, x):
            output, _ = self.lstm(x)
            return self.head(output[:, -1, :])
else:
    RacerPatternLSTM = None


_MODEL = None
_HMM: Optional[LearnedHMM] = None
_active_kind_loaded = False
_active_kind = None


def _load():
    global _MODEL, _HMM, _active_kind_loaded, _active_kind
    if _active_kind_loaded:
        return _MODEL, _HMM
    _active_kind_loaded = True
    _active_kind = None

    if not TORCH_AVAILABLE or not _ACTIVE_MODEL_PATH.exists():
        return None, _load_hmm()

    try:
        model = RacerPatternLSTM()
        state = torch.load(_ACTIVE_MODEL_PATH, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model.eval()
        _MODEL = model
        _active_kind = "lstm"
    except Exception:
        _MODEL = None
    return _MODEL, _load_hmm()


def _load_hmm() -> Optional[LearnedHMM]:
    global _HMM
    if _HMM is not None:
        return _HMM
    if not _ACTIVE_HMM_PATH.exists():
        return None
    try:
        payload = json.loads(_ACTIVE_HMM_PATH.read_text(encoding="utf-8"))
        _HMM = LearnedHMM(payload["transition"])
    except Exception:
        _HMM = None
    return _HMM


def model_is_available() -> bool:
    return bool(TORCH_AVAILABLE and _ACTIVE_MODEL_PATH.exists())


def hmm_is_available() -> bool:
    return _load_hmm() is not None


def active_model_kind() -> Optional[str]:
    _load()
    return _active_kind


def predict_next_tactical(
    window: RivalHistoryWindow,
    fallback_belief: Dict[str, float],
) -> Optional[Dict[str, float]]:
    """Predict next tactical state; None means callers must use safe fallback."""
    model, hmm = _load()
    if model is None or not window.is_ready():
        return None

    with torch.no_grad():
        x = torch.tensor(window.as_list(), dtype=torch.float32).unsqueeze(0)
        probs = torch.softmax(model(x), dim=-1)[0].tolist()
    learned = {s: float(probs[i]) for i, s in enumerate(TACTICAL_STATES)}

    if hmm is not None:
        learned = hmm.predict(learned)

    # Keep the learned layer advisory rather than allowing a model to erase
    # the deterministic observation evidence.
    blend = 0.65
    result = {
        s: blend * learned[s] + (1.0 - blend) * fallback_belief.get(s, 0.0)
        for s in TACTICAL_STATES
    }
    return _normalize(result)
