# ppo_train.py
from __future__ import annotations

import argparse, os
from pathlib import Path
import numpy as np

from gymnasium.wrappers import TimeLimit
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from gym_tetris import TetrisEnv  # your local env


def parse_args():
    p = argparse.ArgumentParser(description="Train MaskablePPO on TetrisEnv")
    p.add_argument("--timesteps", type=int, default=200_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-steps", type=int, default=2000, help="Episode cap")
    p.add_argument("--n-envs", type=int, default=1, help="Parallel envs (try 4–8)")
    p.add_argument("--normalize", action="store_true", help="VecNormalize obs/reward")
    p.add_argument("--save-dir", type=str, default="runs")
    p.add_argument("--save-name", type=str, default="ppo_tetris.zip")
    p.add_argument("--tb-logdir", type=str, default=None)
    return p.parse_args()


def mask_fn(env) -> np.ndarray:
    """
    Unwrap wrappers until we find the base env that exposes valid_action_mask().
    ActionMasker will call this every step.
    """
    base = env
    # Walk down .env chain to find the attribute
    visited = 0
    while not hasattr(base, "valid_action_mask") and hasattr(base, "env") and visited < 10:
        base = base.env
        visited += 1
    if not hasattr(base, "valid_action_mask"):
        # Fallback: allow all actions if something is off
        return np.ones(env.action_space.n, dtype=bool)
    return base.valid_action_mask()


def make_single_env(max_steps: int):
    def _thunk():
        e = TetrisEnv()
        # IMPORTANT: ActionMasker should wrap the BASE env (inner),
        # and TimeLimit should be OUTSIDE (outer) so mask_fn sees TetrisEnv.
        e = ActionMasker(e, mask_fn)
        if max_steps:
            e = TimeLimit(e, max_episode_steps=max_steps)
        return e
    return _thunk


def build_env(args):
    vec_cls = SubprocVecEnv if args.n_envs > 1 else DummyVecEnv
    env = make_vec_env(make_single_env(args.max_steps), n_envs=args.n_envs, seed=args.seed, vec_env_cls=vec_cls)
    if args.normalize:
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    return env


def main():
    args = parse_args()
    outdir = Path(args.save_dir); outdir.mkdir(parents=True, exist_ok=True)

    env = build_env(args)

    model = MaskablePPO(
        "MlpPolicy",
        env,
        seed=args.seed,
        n_steps=2048,
        batch_size=max(512, 256 * args.n_envs),
        learning_rate=2.5e-4,
        ent_coef=0.05,
        gamma=0.995, gae_lambda=0.95,
        clip_range=0.2,
        verbose=1,
        tensorboard_log=args.tb_logdir,
    )

    model.learn(total_timesteps=args.timesteps)

    save_path = outdir / args.save_name
    model.save(str(save_path))
    if isinstance(env, VecNormalize):
        env.save(os.path.join(args.save_dir, "vecnormalize.pkl"))
    print("Saved model to:", save_path)

    env.close()


if __name__ == "__main__":
    main()
