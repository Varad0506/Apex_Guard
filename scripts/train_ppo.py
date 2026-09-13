"""Phase 4: train a PPO policy against ApexGuardEnv (app/rl/apexguard_env.py)
using stable-baselines3.

Requires torch + stable-baselines3, which this build environment doesn't
have disk space to install -- run this in your own environment:

    pip install stable-baselines3[extra] gymnasium

Usage:
    python scripts/train_ppo.py                    # default settings
    python scripts/train_ppo.py --timesteps 500000 --n-envs 16

Key design point carried over from the build guide: PPO ONLY proposes
candidate actions. It is never wired in as the thing that reaches the car.
app/models/policy_adapter.py's propose() function is where this trained
policy plugs in -- see the integration snippet at the bottom of this file's
docstring, and the "Wiring PPO into policy_adapter.py" section in the
README. The verifier (rollout_verified / rollout_fast) always
independently re-checks whatever PPO proposes before anything is selected.
"""

import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from app.rl.apexguard_env import ApexGuardEnv

ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "app" / "data" / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "ppo_policy.zip"
TENSORBOARD_LOG_DIR = Path(__file__).resolve().parents[1] / "app" / "data" / "artifacts" / "ppo_tensorboard"


def build_env_fn(seed: int = 0):
    def _init():
        env = ApexGuardEnv(seed=seed)
        return Monitor(env)
    return _init


def train(
    total_timesteps: int = 50_000,
    n_envs: int = 8,
    eval_freq: int = 10_000,
    seed: int = 42,
):
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    train_env = make_vec_env(
        lambda: ApexGuardEnv(seed=None),  # let each sub-env randomize its own episodes
        n_envs=n_envs,
        seed=seed,
    )
    eval_env = make_vec_env(lambda: ApexGuardEnv(seed=None), n_envs=1, seed=seed + 1000)

    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=3e-4,
        n_steps=1024,          # per-env rollout length before each update
        batch_size=256,
        n_epochs=10,
        gamma=0.995,            # episodes are short (a few ticks), so this matters less than in long-horizon tasks
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,          # encourage exploring all 4 actions early on
        verbose=1,
        tensorboard_log=str(TENSORBOARD_LOG_DIR),
        seed=seed,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(ARTIFACT_DIR / "ppo_best"),
        log_path=str(ARTIFACT_DIR / "ppo_eval_logs"),
        eval_freq=max(eval_freq // n_envs, 1),
        n_eval_episodes=50,
        deterministic=True,
    )

    model.learn(total_timesteps=total_timesteps, callback=eval_callback, progress_bar=True)
    model.save(str(MODEL_PATH))
    print(f"Saved PPO policy -> {MODEL_PATH}")
    print(f"Best checkpoint during training -> {ARTIFACT_DIR / 'ppo_best' / 'best_model.zip'}")
    print(f"TensorBoard logs -> {TENSORBOARD_LOG_DIR}  (run: tensorboard --logdir {TENSORBOARD_LOG_DIR})")

    return model


def quick_evaluate(model_path: Path = MODEL_PATH, n_episodes: int = 200, seed: int = 123):
    """Runs the trained policy against fresh randomized episodes and reports
    pass rate, mean reward, and reserve-breach rate -- the same shape of
    result the guide's evaluation table wants for 'PPO only' performance,
    to compare against the verifier-gated pipeline's numbers from
    scripts/evaluate_rollout.py."""
    model = PPO.load(str(model_path))
    env = ApexGuardEnv(seed=seed)

    passes, breaches, rewards = 0, 0, []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        ep_reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(int(action))
            ep_reward += reward
            done = terminated or truncated
        rewards.append(ep_reward)
        passes += int(env._state.passed)
        breaches += int(env._state.reserve_breached)

    print(f"n_episodes={n_episodes}")
    print(f"pass_rate={passes / n_episodes:.3f}")
    print(f"reserve_breach_rate={breaches / n_episodes:.3f}")
    print(f"mean_reward={sum(rewards) / len(rewards):.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=50_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-only", action="store_true", help="Skip training, just evaluate an existing model")
    args = parser.parse_args()

    if args.eval_only:
        quick_evaluate()
    else:
        train(total_timesteps=args.timesteps, n_envs=args.n_envs, seed=args.seed)
        quick_evaluate()
