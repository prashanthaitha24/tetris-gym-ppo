import numpy as np
from gym_tetris import TetrisEnv

def _to_array(x):
    try:
        return np.array(x)
    except Exception:
        return None

def infer_clears(before_board, after_board, piece_cells=4, row_width=10):
    b = _to_array(before_board); a = _to_array(after_board)
    if b is None or a is None or b.ndim != 2 or a.ndim != 2:
        return 0, 0, 0, 0
    filled_before = int(np.count_nonzero(b))
    filled_after  = int(np.count_nonzero(a))
    numerator = (filled_before + piece_cells) - filled_after
    if row_width <= 0 or numerator < 0 or (numerator % row_width) != 0:
        return 0, filled_before, filled_after, numerator
    clears = numerator // row_width
    return int(max(0, min(clears, 4))), filled_before, filled_after, numerator

def pick_line_clearing_action(env, info):
    before = env.game.board
    row_w = len(before[0]) if before is not None else 10
    next_states = env.game.get_next_states()
    if not next_states:
        return (info.get("valid_actions") or [0])[0], 0
    action_keys = info.get("action_keys") or []
    best_a = (info.get("valid_actions") or [0])[0]
    best_clears = -1
    for local_id in (info.get("valid_actions") or []):
        key = action_keys[local_id] if local_id < len(action_keys) else None
        if key not in next_states:
            continue
        after = next_states[key]
        clears, *_ = infer_clears(before, after, piece_cells=4, row_width=row_w)
        if clears > best_clears:
            best_clears = clears
            best_a = local_id
    return best_a, best_clears

if __name__ == "__main__":
    env = TetrisEnv()
    obs, info = env.reset()
    for t in range(200):
        a, predicted = pick_line_clearing_action(env, info)
        obs, r, term, trunc, info = env.step(a)
        print(f"t={t:03d} a={a} predicted_clears={predicted} | scoreΔ={info.get('score_delta')} linesΔ={info.get('lines_delta')} reward={r:.2f}")
        if term or trunc:
            break
    env.close()
