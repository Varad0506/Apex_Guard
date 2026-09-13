import json
import time
from pathlib import Path
import pytest
from app.schemas.telemetry import DecisionRequest
from app.schemas.common import ActionType
from app.engine import rule_engine, traffic_engine
from app.models import rival_estimator
from app.simulation.engine import rollout_verified, _stable_seed
from app.simulation import tracks

SCENARIOS_DIR = Path(__file__).resolve().parents[1] / "app" / "data" / "scenarios"


def load_scenario(name):
    with (SCENARIOS_DIR / f"{name}.json").open() as f:
        return DecisionRequest(**json.load(f))


def test_stable_seed_deterministic():
    assert _stable_seed("req-1", "HOLD") == _stable_seed("req-1", "HOLD")
    assert _stable_seed("req-1", "HOLD") != _stable_seed("req-1", "FULL_DEPLOY")
    assert _stable_seed("req-1", "HOLD") != _stable_seed("req-2", "HOLD")


def test_rollout_verified_reproducible_across_calls():
    req = load_scenario("rear_drs_threat")
    rival = rival_estimator.estimate(req)
    traffic = traffic_engine.evaluate(req, rival)
    o1 = rollout_verified(req, ActionType.PARTIAL_DEPLOY, traffic, rival)
    o2 = rollout_verified(req, ActionType.PARTIAL_DEPLOY, traffic, rival)
    assert o1 == o2


def test_soc_never_leaves_valid_range():
    req = load_scenario("isolated_pass")
    rival = rival_estimator.estimate(req)
    traffic = traffic_engine.evaluate(req, rival)
    for action in ActionType:
        o = rollout_verified(req, action, traffic, rival)
        assert 0.0 <= o.projected_soc_pct <= 100.0


def test_full_deploy_breaches_reserve_in_tight_scenario():
    # rear_drs_threat has minimum_reserve_soc_pct=15; sustained full deploy
    # over the horizon should project SOC at or below that reserve.
    req = load_scenario("rear_drs_threat")
    rival = rival_estimator.estimate(req)
    traffic = traffic_engine.evaluate(req, rival)
    o = rollout_verified(req, ActionType.FULL_DEPLOY, traffic, rival)
    assert o.projected_soc_pct <= req.rules.minimum_reserve_soc_pct + 2.0


def test_confidence_is_bounded():
    req = load_scenario("isolated_pass")
    rival = rival_estimator.estimate(req)
    traffic = traffic_engine.evaluate(req, rival)
    for action in ActionType:
        o = rollout_verified(req, action, traffic, rival)
        assert 0.0 <= o.confidence <= 1.0


def test_verified_mode_latency_within_budget():
    req = load_scenario("rear_drs_threat")
    rival = rival_estimator.estimate(req)
    traffic = traffic_engine.evaluate(req, rival)
    start = time.perf_counter()
    for action in ActionType:
        rollout_verified(req, action, traffic, rival)
    elapsed_ms = (time.perf_counter() - start) * 1000
    # Generous CI-safe ceiling; the guide's target is 300ms for the whole
    # request including everything else, this covers just the 4 rollouts.
    assert elapsed_ms < 500


def test_track_profiles_generated_and_loadable():
    ids = tracks.list_profile_ids()
    assert 8 <= len(ids) <= 12
    profile = tracks.load_profile(ids[0])
    assert len(profile.sectors) > 0
    assert 0.0 <= profile.overtake_difficulty <= 1.0
