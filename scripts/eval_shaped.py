import os, warnings
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message="Protobuf gencode version")

import argparse
from statistics import mean, median
from stable_baselines3 import PPO
from gym_tetris import TetrisEnv

def run_episode(env, model, deterministic=False, max_steps=5000):
    obs, info = env.reset()
    steps = lines = score = 0
    while steps < max_steps:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, r, term, trunc, info = env.step(int(action))
        lines += int(info.get("lines_delta", 0))
        score += int(info.get("score_delta", 0))
        steps += 1
        if term or trunc:
            break
    return steps, lines, score

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="runs/ppo_tetris_shaped.zip")
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--deterministic", action="store_true")
    args = p.parse_args()

    env = TetrisEnv()
    model = PPO.load(args.model)

    results = [run_episode(env, model, deterministic=args.deterministic) for _ in range(args.episodes)]
    env.close()

    steps = [r[0] for r in results]
    lines = [r[1] for r in results]
    scores = [r[2] for r in results]

    print({
        "episodes": args.episodes,
        "avg_steps": round(mean(steps), 2),
        "median_steps": int(median(steps)),
        "avg_lines": round(mean(lines), 2),     
        "avg_score": round(mean(scores), 2),
    })
