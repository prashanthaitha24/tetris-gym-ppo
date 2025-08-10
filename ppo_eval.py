# ppo_eval.py
import argparse
from pathlib import Path
import numpy as np
from stable_baselines3 import PPO
from gym_tetris import TetrisEnv


def to_scalar(a):
    return int(a.squeeze()) if isinstance(a, np.ndarray) else int(a)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True, help="Path to PPO .zip (e.g., runs/ppo_tetris.zip)")
    p.add_argument("--episodes", type=int, default=3, help="Number of eval episodes")
    p.add_argument("--deterministic", action="store_true", help="Deterministic policy actions")
    args = p.parse_args()

    # Create env (Gymnasium-style) and load model (no env passed to avoid space checks)
    env = TetrisEnv()
    model = PPO.load(str(args.model))

    for ep in range(1, args.episodes + 1):
        obs, info = env.reset()
        print(f"Episode {ep}: valid_actions at reset = {info.get('valid_actions', 'n/a')}")
        done = False
        total_r = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=args.deterministic)
            action = to_scalar(action)
            obs, r, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            total_r += float(r)
        print(f"Episode {ep}: reward={total_r:.2f} score={info.get('score', 0)} lines={info.get('lines', 0)}")

    env.close()


if __name__ == "__main__":
    main()
