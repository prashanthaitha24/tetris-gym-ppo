from __future__ import annotations

import argparse
import statistics as stats
from pathlib import Path
import numpy as np

from gymnasium.wrappers import TimeLimit
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from gym_tetris import TetrisEnv


def mask_fn(env) -> np.ndarray:
    # find env.valid_action_mask()
    base = env
    hops = 0
    while not hasattr(base, "valid_action_mask") and hasattr(base, "env") and hops < 10:
        base = base.env
        hops += 1
    if hasattr(base, "valid_action_mask"):
        return base.valid_action_mask()
    # Fallback: allow all actions
    return np.ones(env.action_space.n, dtype=bool)


def make_env(max_steps: int):
    def _thunk():
        e = TetrisEnv()
        e = ActionMasker(e, mask_fn)             # ensure masks available
        if max_steps:
            e = TimeLimit(e, max_episode_steps=max_steps)
        return e
    return _thunk


def run_episode_vec(vec_env, model, deterministic: bool = False):
    """
    Run exactly ONE episode on a 1-env DummyVecEnv.
    Returns dict with steps, reward, lines, topout.
    """
    # VecEnv reset
    obs = vec_env.reset()
    steps = 0
    total_r = 0.0
    lines = 0
    topout = False

    while True:
        # model.predict expects batched obs from VecEnv
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, done, infos = vec_env.step(action)

        steps += 1
        total_r += float(reward[0])
        # infos is a list; we have 1 env -> infos[0]
        lines += int(infos[0].get("lines_delta", 0) or 0)

        if done[0]:
            # We stored topout (terminated) in info at step() time if we choose to add it;
            # otherwise treat any 'done' as episode end.
            topout = bool(infos[0].get("terminal_observation") is not None or infos[0].get("final_observation") is not None)
            break

    return {"steps": steps, "reward": total_r, "lines": lines, "topout": topout}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="runs/ppo_tetris.zip")
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--max-steps", type=int, default=2000)
    ap.add_argument("--save-dir", type=str, default="runs", help="where vecnormalize.pkl may live")
    ap.add_argument("--deterministic", action="store_true", help="Use deterministic policy at eval")
    args = ap.parse_args()

    # Build single env for evaluation
    env = DummyVecEnv([make_env(args.max_steps)])

    # If we train with --normalize, load stats so obs/reward scaling matches training
    vn_path = Path(args.save_dir) / "vecnormalize.pkl"
    if vn_path.exists():
        env = VecNormalize.load(str(vn_path), env)
        env.training = False
        env.norm_reward = False  # No normalize rewards at eval time

    # Load model (MaskablePPO)
    model = MaskablePPO.load(args.model)

    results = [run_episode_vec(env, model, deterministic=args.deterministic) for _ in range(args.episodes)]
    env.close()

    steps = [r["steps"] for r in results]
    lines = [r["lines"] for r in results]
    topouts = sum(1 for r in results if r["topout"])
    print({
        "episodes": args.episodes,
        "avg_steps": round(stats.mean(steps), 2),
        "median_steps": int(stats.median(steps)),
        "avg_lines": round(stats.mean(lines), 3),
        "topouts": topouts
    })
