"""Cheap, always-on check: is an overtake window plausibly forming?
Only when this flags a candidate does the heavier pipeline run."""

from app.schemas.telemetry import DecisionRequest


def score(req: DecisionRequest) -> float:
    """Returns an opportunity score in [0, 1] from gap, closing speed and
    DRS/segment availability. Pure heuristic, no learned component."""
    target = req.target
    track = req.track

    # Closing speed component: positive relative speed means we're catching up.
    closing = max(0.0, target.relative_speed_kph) / 40.0  # normalize ~40kph as strong closing
    closing_score = min(1.0, closing)

    # Gap component: smaller gap = higher opportunity, capped at ~2s relevance window.
    gap_score = max(0.0, 1.0 - (target.gap_s / 2.0))

    # DRS/segment bonus.
    drs_bonus = 0.25 if (track.drs_available and track.segment_type == "DRS_STRAIGHT") else 0.0

    # Track difficulty dampens opportunity on hard-to-pass circuits.
    difficulty_penalty = track.overtake_difficulty * 0.2

    raw = 0.45 * gap_score + 0.35 * closing_score + drs_bonus - difficulty_penalty
    return max(0.0, min(1.0, raw))
