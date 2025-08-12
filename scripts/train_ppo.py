# scripts/train_ppo.py
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Callable
import numpy as np

from gymnasium.wrappers import TimeLimit

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback

from gym_tetris import TetrisEnv


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train MaskablePPO on TetrisEnv")
    p.add_argument("--timesteps", type=int, default=200_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-steps", type=int, default=2000, help="Episode cap via TimeLimit")
    p.add_argument("--n-envs", type=int, default=1, help="Parallel envs (try 4–8 for speed)")
    p.add_argument("--normalize", action="store_true", help="Use VecNormalize (obs/reward)")
    p.add_argument("--save-dir", type=str, default="runs", help="Where to save models/logs")
    p.add_argument("--save-name", type=str, default="ppo_tetris.zip")
    p.add_argument("--tb-logdir", type=str, default=None, help="TensorBoard log dir")
    p.add_argument("--ent-coef", type=float, default=0.05, help="Entropy bonus (exploration)")
    return p.parse_args()


def mask_fn(env) -> np.ndarray:
    """
    Return boolean action mask; unwrap wrappers until we find base env.valid_action_mask().
    ActionMasker calls this each step.
    """
    base = env
    for _ in range(10):
        if hasattr(base, "valid_action_mask"):
            break
        if hasattr(base, "env"):
            base = base.env
    if hasattr(base, "valid_action_mask"):
        return base.valid_action_mask()
    return np.ones(env.action_space.n, dtype=bool)  # fallback: allow all


def make_single_env(max_steps: int) -> Callable[[], TetrisEnv]:
    def _thunk():
        e = TetrisEnv()
        # Mask inner, then cap episode length outside
        e = ActionMasker(e, mask_fn)
        if max_steps:
            e = TimeLimit(e, max_episode_steps=max_steps)
        return e
    return _thunk


def build_train_env(args: argparse.Namespace):
    vec_cls = SubprocVecEnv if args.n_envs > 1 else DummyVecEnv
    env = make_vec_env(make_single_env(args.max_steps), n_envs=args.n_envs, seed=args.seed, vec_env_cls=vec_cls)
    if args.normalize:
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    return env


def build_eval_env(args: argparse.Namespace):
    # single-env eval with identical wrappers
    eval_env = DummyVecEnv([make_single_env(args.max_steps)])
    if args.normalize:
        eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True, clip_obs=10.0)
        eval_env.training = False
        eval_env.norm_reward = False
    return eval_env


def main():
    args = parse_args()
    outdir = Path(args.save_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Build envs
    env = build_train_env(args)
    eval_env = build_eval_env(args)

    # Model (good exploration defaults)
    model = MaskablePPO(
        "MlpPolicy",
        env,
        seed=args.seed,
        n_steps=2048,
        batch_size=max(512, 256 * args.n_envs),
        learning_rate=2.5e-4,
        ent_coef=args.ent_coef,   # exploration
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1,
        tensorboard_log=args.tb_logdir,
    )

    # If using VecNormalize, copy running stats to eval env (NO .load() here)
    if args.normalize and isinstance(env, VecNormalize) and isinstance(eval_env, VecNormalize):
        eval_env.obs_rms = env.obs_rms
        eval_env.ret_rms = env.ret_rms
        eval_env.training = False
        eval_env.norm_reward = False

    # Eval callback (saves best model)
    best_dir = outdir / "best"
    best_dir.mkdir(parents=True, exist_ok=True)
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(best_dir),
        log_path=str(outdir / "eval"),
        eval_freq=max(1, 10_000 // max(1, args.n_envs)),  # ~every 10k steps
        deterministic=True,
        render=False,
    )

    # Train
    model.learn(total_timesteps=args.timesteps, callback=eval_cb)

    # Save final model + (optional) VecNormalize stats for offline eval
    final_path = outdir / args.save_name
    model.save(str(final_path))
    if args.normalize and isinstance(env, VecNormalize):
        env.save(str(outdir / "vecnormalize.pkl"))

    print("Saved final model:", final_path)
    print("Best model (EvalCallback):", best_dir / "best_model.zip")

    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
