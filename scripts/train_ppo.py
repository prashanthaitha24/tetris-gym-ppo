# scripts/train_ppo.py
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from gym_tetris import TetrisEnv

def make_env():
    return TetrisEnv()

class RewardShapedTetris(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.last_board = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_board = obs.copy()
        return obs, info

    def step(self, action):
        obs, _, done, trunc, info = self.env.step(action)

        lines_cleared = info.get("lines_delta", 0)
        score_delta = info.get("score_delta", 0)

        shaped = 0.0
        shaped += lines_cleared * 2.0     # strong positive
        shaped += score_delta * 1.0       # soft positive
        shaped += 0.02                    # survival bonus

        if self.last_board is not None:
            holes_before = self._holes(self.last_board)
            holes_after = self._holes(obs)
            shaped -= max(0, holes_after - holes_before) * 0.5

            h_before = self._max_height(self.last_board)
            h_after = self._max_height(obs)
            shaped -= max(0, h_after - h_before) * 0.05

        self.last_board = obs.copy()
        return obs, float(shaped), done, trunc, info

    def _holes(self, board):
        H, W = board.shape
        filled = board != 0
        holes = 0
        for c in range(W):
            seen = False
            for r in range(H):
                if filled[r, c]:
                    seen = True
                elif seen:
                    holes += 1
        return holes

    def _max_height(self, board):
        H, W = board.shape
        filled = board != 0
        h = 0
        for c in range(W):
            col = filled[:, c]
            h = max(h, H - np.argmax(col) if np.any(col) else 0)
        return h

if __name__ == "__main__":
    env = DummyVecEnv([lambda: RewardShapedTetris(make_env())])

    model = PPO(
        "MlpPolicy",
        env,
        n_steps=2048,
        batch_size=256,
        learning_rate=3e-4,
        ent_coef=0.01,
        clip_range=0.2,
        verbose=1,
        tensorboard_log="./ppo_tetris_tensorboard/"
    )

    timesteps = int(os.environ.get("TIMESTEPS", 200_000))
    model.learn(total_timesteps=timesteps)
    model.save("runs/ppo_tetris_shaped")

    print("✅ Training complete. Model saved to runs/ppo_tetris_shaped.zip")
