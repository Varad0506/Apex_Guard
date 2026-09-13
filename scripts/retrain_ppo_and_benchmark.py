"""Retrain the 27-D confidence-aware PPO policy and benchmark it against non-PPO.

Run from the repository root:
    python -m scripts.retrain_ppo_and_benchmark --timesteps 50000 --cases 80

This intentionally retrains from scratch because the observation space changed
from 24 to 27 dimensions. The same verifier stack is used for both benchmark
variants; PPO remains proposal-only and cannot bypass the verifier.
"""
from __future__ import annotations

import argparse
from scripts import train_ppo
from scripts import benchmark_ppo_vs_nonppo


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps", type=int, default=50_000)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cases", type=int, default=80)
    args = p.parse_args()

    print(f"[1/3] Training confidence-aware PPO: {args.timesteps} timesteps")
    train_ppo.train(total_timesteps=args.timesteps, n_envs=args.n_envs, seed=args.seed)

    print("[2/3] Evaluating trained PPO")
    train_ppo.quick_evaluate(n_episodes=200, seed=123)

    print(f"[3/3] Benchmarking PPO vs non-PPO on {args.cases} cases")
    benchmark_ppo_vs_nonppo.main(args.cases)


if __name__ == "__main__":
    main()
