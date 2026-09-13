import json
from pathlib import Path
from app.schemas.telemetry import DecisionRequest
from app.engine import rule_engine
from app.models import policy_adapter

SCENARIOS_DIR = Path(__file__).resolve().parents[1] / "app" / "data" / "scenarios"
PPO_MODEL_PATH = Path(__file__).resolve().parents[1] / "app" / "data" / "artifacts" / "ppo_policy.zip"


def load_scenario(name):
    with (SCENARIOS_DIR / f"{name}.json").open() as f:
        return DecisionRequest(**json.load(f))


def test_propose_returns_none_without_trained_artifact():
    # This build has no ppo_policy.zip and no stable_baselines3 installed --
    # propose() must degrade to "no proposal" rather than raise.
    req = load_scenario("isolated_pass")
    legal = rule_engine.allowed_actions(req)
    result = policy_adapter.propose(req, legal)
    if not PPO_MODEL_PATH.exists():
        assert result is None


def test_policy_is_available_matches_artifact_presence():
    available = policy_adapter.policy_is_available()
    if not PPO_MODEL_PATH.exists():
        assert available is False


def test_observation_vector_dimension_matches_env():
    from app.rl.apexguard_env import OBS_DIM
    req = load_scenario("rear_drs_threat")
    obs = policy_adapter._observation_from_request(req)
    assert obs.shape == (OBS_DIM,)


def test_action_ordering_matches_env():
    from app.rl.apexguard_env import ACTIONS
    assert policy_adapter._ACTIONS == ACTIONS
