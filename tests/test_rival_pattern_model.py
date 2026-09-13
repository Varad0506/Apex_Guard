import json

from app.models import rival_pattern_model
from app.models.rival_pattern_model import (
    RivalHistoryWindow,
    LearnedHMM,
    TACTICAL_STATES,
    make_feature_vector,
    proxy_label,
)


def test_feature_vector_is_bounded():
    f = make_feature_vector(gap_s=1.0, relative_speed_kph=-30.0, sector_kind="DRS_STRAIGHT", dt_s=5.0)
    assert len(f) == rival_pattern_model.FEATURE_DIM
    assert all(-1.0 <= v <= 1.0 for v in f)


def test_proxy_label_picks_argmax():
    belief = {"HARVESTING": 0.1, "DEFENDING": 0.6, "ATTACKING": 0.2, "CONSERVING": 0.1}
    assert TACTICAL_STATES[proxy_label(belief)] == "DEFENDING"


def test_history_window_ready_only_after_enough_pushes():
    w = RivalHistoryWindow()
    assert not w.is_ready()
    for _ in range(rival_pattern_model.WINDOW_SIZE):
        w.push(gap_s=1.0, relative_speed_kph=0.0, sector_kind="CORNER", dt_s=0.5)
    assert w.is_ready()
    assert len(w.as_list()) == rival_pattern_model.WINDOW_SIZE


def test_learned_hmm_predict_is_a_valid_distribution():
    n = len(TACTICAL_STATES)
    transition = [[1.0 / n] * n for _ in range(n)]
    hmm = LearnedHMM(transition)
    previous = {s: 1.0 / n for s in TACTICAL_STATES}
    predicted = hmm.predict(previous)
    assert set(predicted) == set(TACTICAL_STATES)
    assert abs(sum(predicted.values()) - 1.0) < 1e-6


def test_no_active_model_returns_none_and_callers_fall_back(tmp_path, monkeypatch):
    monkeypatch.setattr(rival_pattern_model, "_ACTIVE_MODEL_PATH", tmp_path / "missing.json")
    monkeypatch.setattr(rival_pattern_model, "_active_kind_loaded", False)
    monkeypatch.setattr(rival_pattern_model, "_active_kind", None)
    rival_pattern_model._MODEL = None
    assert rival_pattern_model.model_is_available() is False
    window = RivalHistoryWindow()
    for _ in range(rival_pattern_model.WINDOW_SIZE):
        window.push(gap_s=1.0, relative_speed_kph=0.0, sector_kind="CORNER", dt_s=0.5)
    result = rival_pattern_model.predict_next_tactical(window, {s: 0.25 for s in TACTICAL_STATES})
    assert result is None


def test_opponent_belief_still_works_with_no_learned_model():
    from app.schemas.telemetry import DecisionRequest
    from app.models import rival_estimator, opponent_belief
    from pathlib import Path

    frame = json.loads((Path(__file__).resolve().parents[1]
                        / "app" / "data" / "scenarios" / "isolated_pass.json").read_text())
    req = DecisionRequest.model_validate(frame)
    rival = rival_estimator.estimate(req)

    belief_filter = opponent_belief.OpponentBeliefFilter()
    state1 = belief_filter.update(req, rival)
    state2 = belief_filter.update(req, rival)

    assert abs(sum(state2.tactical_belief.values()) - 1.0) < 1e-6
    assert state2.update_count == 2
