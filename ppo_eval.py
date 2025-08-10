from stable_baselines3 import PPO
from gym_tetris import TetrisEnv
import argparse, pathlib
import numpy as np

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--model', type=pathlib.Path, required=True)
    p.add_argument('--episodes', type=int, default=3)
    a = p.parse_args()

    env = TetrisEnv()
    model = PPO.load(str(a.model))

    for ep in range(a.episodes):
        obs, info = env.reset()
        done, total = False, 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            # >>> cast to scalar int <<<
            if isinstance(action, np.ndarray):
                action = int(action.squeeze())
            else:
                action = int(action)
            obs, r, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total += float(r)
        print(f"Episode {ep+1}: reward={total} score={info.get('score')}")
