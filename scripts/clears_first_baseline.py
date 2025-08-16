import os, warnings
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message="Protobuf gencode version")

import numpy as np
from statistics import mean, median
from gym_tetris import TetrisEnv

BOARD_H, BOARD_W = 20, 10

def _to_board2d(x):
    try:
        b = np.array(x)
    except Exception:
        return np.zeros((BOARD_H, BOARD_W), dtype=np.uint8)
    if b.ndim == 1:
        if b.size == BOARD_H * BOARD_W:
            b = b.reshape(BOARD_H, BOARD_W)
        else:
            return np.zeros((BOARD_H, BOARD_W), dtype=np.uint8)
    elif b.ndim != 2:
        return np.zeros((BOARD_H, BOARD_W), dtype=np.uint8)
    return (b != 0).astype(np.uint8)

def infer_clears(before_board, after_board, row_width=10):
    b = _to_board2d(before_board); a = _to_board2d(after_board)
    filled_before = int(np.count_nonzero(b))
    filled_after  = int(np.count_nonzero(a))
    numerator = (filled_before + 4) - filled_after
    if row_width <= 0 or numerator < 0 or (numerator % row_width) != 0:
        return 0
    return int(max(0, min(numerator // row_width, 4)))

def pick_action(env, info):
    before = _to_board2d(env.game.board)
    row_w = before.shape[1]
    next_states = env.game.get_next_states()
    if not next_states:
        return (info.get("valid_actions") or [0])[0]
    action_keys = info.get("action_keys") or []
    best_a = (info.get("valid_actions") or [0])[0]
    best_clears = -1
    for a in (info.get("valid_actions") or []):
        key = action_keys[a] if a < len(action_keys) else None
        if key not in next_states:
            continue
        clears = infer_clears(before, next_states[key], row_width=row_w)
        if clears > best_clears:
            best_clears = clears
            best_a = a
    return best_a

def run_episode(env: TetrisEnv, max_steps=5000):
    obs, info = env.reset()
    steps = 0
    total_lines = 0
    while steps < max_steps:
        a = pick_action(env, info)
        obs, r, term, trunc, info = env.step(a)
        total_lines += int(info.get("lines_delta", 0))
        steps += 1
        if term or trunc:
            break
    return {"steps": steps, "lines": total_lines}

if __name__ == "__main__":
    env = TetrisEnv()
    episodes = 20
    results = [run_episode(env) for _ in range(episodes)]
    env.close()
    print({
        "episodes": episodes,
        "avg_steps": round(mean([r["steps"] for r in results]), 2),
        "median_steps": int(median([r["steps"] for r in results])),
        "avg_lines": round(mean([r["lines"] for r in results]), 2),
    })
