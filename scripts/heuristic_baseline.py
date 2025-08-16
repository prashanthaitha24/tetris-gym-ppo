import os, warnings
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
# Silenced some generic user warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message="Protobuf gencode version")

import numpy as np
from statistics import mean, median
from gym_tetris import TetrisEnv

BOARD_H, BOARD_W = 20, 10  # expected grid

def _to_board2d(x):
    """Coerce any engine board into a 2D (H,W) numpy array of 0/1."""
    try:
        b = np.array(x)
    except Exception:
        return np.zeros((BOARD_H, BOARD_W), dtype=np.uint8)

    # If 1D of size H*W, reshape; else try to infer
    if b.ndim == 1:
        if b.size == BOARD_H * BOARD_W:
            b = b.reshape(BOARD_H, BOARD_W)
        else:
            # best effort: fill zeros
            return np.zeros((BOARD_H, BOARD_W), dtype=np.uint8)
    elif b.ndim != 2:
        return np.zeros((BOARD_H, BOARD_W), dtype=np.uint8)

    # Binarize occupancy
    return (b != 0).astype(np.uint8)

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
    return int(holes)

def heights(board):
    H, W = board.shape
    filled = board != 0
    hs = np.zeros(W, dtype=int)
    for c in range(W):
        col = filled[:, c]
        hs[c] = H - np.argmax(col) if np.any(col) else 0
    return hs

def bumpiness(hs):
    return int(np.sum(np.abs(np.diff(hs))))

def full_rows(board):
    return int(np.sum(np.all(board != 0, axis=1)))

def score_state(board_like):
    """Higher is better (we’ll argmax)."""
    b = _to_board2d(board_like)
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
            total_reward += float(r)
            break

        keys = list(next_states.keys())
        # conversion to 2D
        boards = [_to_board2d(next_states[k]) for k in keys]
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
