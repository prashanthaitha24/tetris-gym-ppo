from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from gym_tetris import TetrisEnv
import argparse, pathlib

def make_env():
    return TetrisEnv()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--timesteps', type=int, default=200_000)
    p.add_argument('--outdir', type=pathlib.Path, default=pathlib.Path('runs'))
    a = p.parse_args()

    a.outdir.mkdir(parents=True, exist_ok=True)
    env = DummyVecEnv([make_env])

    model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=str(a.outdir / 'tb'))
    model.learn(total_timesteps=a.timesteps)
    model.save(str(a.outdir / 'ppo_tetris'))
    print('Saved:', a.outdir / 'ppo_tetris.zip')
