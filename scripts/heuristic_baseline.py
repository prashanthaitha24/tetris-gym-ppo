from gym_tetris import TetrisEnv
import statistics as stats
import numpy as np

def choose_action(info, action_space):
    va = info.get("valid_actions")
    if isinstance(va, (list, tuple)) and va:
        # naive: prefer “do nothing / soft drop” if present, else random valid
        # tweak this mapping to your env’s action ids
        priority = [0, 1, 2, 3, 4]  # example placeholders
        for p in priority:
            if p in va: return p
        return int(np.random.choice(va))
    return action_space.sample()

def run_episode(env):
    obs, info = env.reset()
    steps = lines = 0
    while True:
        a = choose_action(info, env.action_space)
        obs, r, term, trunc, info = env.step(a)
        steps += 1
        lines += int(info.get("lines_delta", 0) or 0)
        if term or trunc: break
    return steps, lines, term

def quick_bench(n=50):
    env = TetrisEnv()
    res = [run_episode(env) for _ in range(n)]
    env.close()
    steps=[x[0] for x in res]; lines=[x[1] for x in res]; died=[x[2] for x in res]
    return {
        "episodes": n,
        "avg_steps": round(stats.mean(steps),2),
        "median_steps": int(stats.median(steps)),
        "avg_lines": round(stats.mean(lines),2),
        "topouts": sum(died),
    }

if __name__ == "__main__":
    print(quick_bench(50))
