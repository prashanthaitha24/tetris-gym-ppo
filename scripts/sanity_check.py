from gym_tetris import TetrisEnv
import numpy as np
import statistics as stats

def run_episode(env, seed=None):
    if seed is not None:
        try:
            env.reset(seed=seed)
        except TypeError:
            pass
    obs, info = env.reset()
    steps = 0
    total_r = 0.0
    total_lines = 0
    died = False
    last_info = info

    while True:
        # Only valid actions
        if isinstance(last_info.get("valid_actions"), (list, tuple)) and last_info["valid_actions"]:
            action = np.random.choice(last_info["valid_actions"])
        else:
            action = env.action_space.sample()

        obs, r, terminated, truncated, info = env.step(action)
        steps += 1
        total_r += float(r)
        total_lines += int(info.get("lines_delta", 0) or 0)

        if terminated or truncated:
            died = terminated and not truncated
            return {
                "steps": steps,
                "reward": total_r,
                "lines": total_lines,
                "died": died
            }

        last_info = info

def quick_benchmark(n=20):
    env = TetrisEnv()
    results = [run_episode(env) for _ in range(n)]
    env.close()
    steps = [r["steps"] for r in results]
    lines = [r["lines"] for r in results]
    return {
        "episodes": n,
        "avg_steps": round(stats.mean(steps), 2),
        "median_steps": int(stats.median(steps)),
        "avg_lines": round(stats.mean(lines), 2),
        "topouts": sum(r["died"] for r in results)
    }

if __name__ == "__main__":
    stats_out = quick_benchmark(25)
    print(stats_out)
