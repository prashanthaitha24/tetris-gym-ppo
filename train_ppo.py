from gym_tetris import TetrisEnv
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO

def make_env():
    e = TetrisEnv()
    e = TimeLimit(e, max_episode_steps=2000)   # prevent ultra-long episodes
    return e

if __name__ == "__main__":
    env = make_env()
    print("obs:", env.observation_space, "act:", env.action_space)
    model = PPO("MlpPolicy", env,
                n_steps=2048, batch_size=256, learning_rate=3e-4,
                ent_coef=0.01, clip_range=0.2, verbose=1)
    model.learn(total_timesteps=200_000)       # smoke test
    model.save("ppo_tetris_smoke")
    env.close()
