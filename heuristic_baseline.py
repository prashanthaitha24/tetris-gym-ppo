import numpy as np
from statistics import mean, median
from gym_tetris import TetrisEnv

def holes(board):
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

def heights(board):
    H, W = board.shape
    filled = board != 0
    hs = np.zeros(W, dtype=int)
    for c in range(W):
        col = filled[:, c]
        if np.any(col):
            hs[c] = H - np.argmax(col)
        else:
            hs[c] = 0
    return hs

def bumpiness(hs):
    return int(np.sum(np.abs(np.diff(hs))))

def full_rows(board):
    return int(np.sum(np.all(board != 0, axis=1)))

def score_state(board_like):
    """Higher is better (we’ll argmax)."""
    b = np.array(board_like)
    hs = heights(b)
    return (
        3.0 * full_rows(b)          # clear lines is very good
        - 2.0 * holes(b)            # penalize holes
        - 0.5 * np.sum(hs)          # lower total stack is better
        - 0.3 * bumpiness(hs)       # smoother surface is better
    )

def run_episode(env: TetrisEnv, max_steps=5000):
    obs, info = env.reset()
    steps = 0
    total_reward = 0.0
    total_lines = 0
    scoring_steps = 0

    while steps < max_steps:
        next_states = env.game.get_next_states()
        if not next_states:
            _, r, term, trunc, info = env.step(0)
            total_reward += r
            break

        keys = list(next_states.keys())
        boards = [np.array(next_states[k]) for k in keys]
        scores = [score_state(b) for b in boards]
        best_idx = int(np.argmax(scores))
        best_key = keys[best_idx]

        if "action_keys" not in info or not info["action_keys"]:
            a = (info.get("valid_actions") or [0])[0]
        else:
            action_keys = info["action_keys"]
            try:
                a = int(action_keys.index(best_key))
            except ValueError:
                a = (info.get("valid_actions") or [0])[0]

        obs, r, term, trunc, info = env.step(a)
        total_reward += float(r)
        total_lines += int(info.get("lines_delta", 0))
        scoring_steps += int(info.get("score_delta", 0) > 0)

        steps += 1
        if term or trunc:
            break

    return {
        "steps": steps,
        "reward": total_reward,
        "lines": total_lines,
        "scoring_steps": scoring_steps
    }

if __name__ == "__main__":
    env = TetrisEnv()
    episodes = 50
    results = [run_episode(env) for _ in range(episodes)]
    env.close()

    avg_steps = mean([r["steps"] for r in results])
    med_steps = median([r["steps"] for r in results])
    avg_lines = mean([r["lines"] for r in results])
    avg_scoring_steps = mean([r["scoring_steps"] for r in results])

    print({
        "episodes": episodes,
        "avg_steps": round(avg_steps, 2),
        "median_steps": med_steps,
        "avg_lines": round(avg_lines, 2),
        "avg_scoring_steps": round(avg_scoring_steps, 2)
    })
