import os, warnings, argparse
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message="Protobuf gencode version")

from statistics import mean, median
import numpy as np
from stable_baselines3 import PPO
from gym_tetris import TetrisEnv

BOARD_H, BOARD_W = 20, 10

def _to_board2d(x):
    try:
        b = np.array(x)
    except Exception:
        return np.zeros((BOARD_H, BOARD_W), dtype=np.uint8)
    if b.ndim == 1 and b.size == BOARD_H * BOARD_W:
        b = b.reshape(BOARD_H, BOARD_W)
    if b.ndim != 2:
        return np.zeros((BOARD_H, BOARD_W), dtype=np.uint8)
    return (b != 0).astype(np.uint8)

# ---- Baseline 1 ----
def _infer_clears(before_board, after_board, row_width=10):
    b = _to_board2d(before_board); a = _to_board2d(after_board)
    filled_before = int(np.count_nonzero(b))
    filled_after  = int(np.count_nonzero(a))
    numerator = (filled_before + 4) - filled_after
    if row_width <= 0 or numerator < 0 or (numerator % row_width) != 0:
        return 0
    return int(max(0, min(numerator // row_width, 4)))

def _pick_action_clears_first(env, info):
    before = _to_board2d(env.game.board)
    row_w = before.shape[1]
    next_states = env.game.get_next_states()
    if not next_states:
        return (info.get("valid_actions") or [0])[0]
    action_keys = info.get("action_keys") or []
    best_a, best_clears = (info.get("valid_actions") or [0])[0], -1
    for a in (info.get("valid_actions") or []):
        key = action_keys[a] if a < len(action_keys) else None
        if key not in next_states:
            continue
        clears = _infer_clears(before, next_states[key], row_width=row_w)
        if clears > best_clears:
            best_clears, best_a = clears, a
    return best_a

def run_episode_clears_first(env, max_steps=5000):
    obs, info = env.reset()
    steps = lines = score = 0
    while steps < max_steps:
        a = _pick_action_clears_first(env, info)
        obs, r, term, trunc, info = env.step(a)
        lines += int(info.get("lines_delta", 0))
        score += int(info.get("score_delta", 0))
        steps += 1
        if term or trunc:
            break
    return steps, lines, score

# ---- Baseline 2: Heuristic surface/holes ----
def _holes(board):
    H, W = board.shape
    filled = board != 0
    holes = 0
    for c in range(W):
        seen = False
        for r in range(H):
            if filled[r, c]: seen = True
            elif seen: holes += 1
    return int(holes)

def _heights(board):
    H, W = board.shape
    filled = board != 0
    hs = np.zeros(W, dtype=int)
    for c in range(W):
        col = filled[:, c]
        hs[c] = H - np.argmax(col) if np.any(col) else 0
    return hs

def _bumpiness(hs): return int(np.sum(np.abs(np.diff(hs))))
def _full_rows(board): return int(np.sum(np.all(board != 0, axis=1)))

def _score_state(board_like):
    b = _to_board2d(board_like)
    hs = _heights(b)
    return (
        3.0 * _full_rows(b)
        - 2.0 * _holes(b)
        - 0.5 * np.sum(hs)
        - 0.3 * _bumpiness(hs)
    )

def _pick_action_heuristic(env, info):
    next_states = env.game.get_next_states()
    if not next_states:
        return (info.get("valid_actions") or [0])[0]
    keys = list(next_states.keys())
    boards = [_to_board2d(next_states[k]) for k in keys]
    scores = [_score_state(b) for b in boards]
    best_key = keys[int(np.argmax(scores))]
    action_keys = info.get("action_keys") or []
    try:
        a = int(action_keys.index(best_key))
    except Exception:
        a = (info.get("valid_actions") or [0])[0]
    return a

def run_episode_heuristic(env, max_steps=5000):
    obs, info = env.reset()
    steps = lines = score = 0
    while steps < max_steps:
        a = _pick_action_heuristic(env, info)
        obs, r, term, trunc, info = env.step(a)
        lines += int(info.get("lines_delta", 0))
        score += int(info.get("score_delta", 0))
        steps += 1
        if term or trunc:
            break
    return steps, lines, score

# ---- PPO ----
def run_episode_ppo(env, model, deterministic=False, max_steps=5000):
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

def summarize(name, triples):
    steps = [t[0] for t in triples]
    lines = [t[1] for t in triples]
    scores= [t[2] for t in triples]
    return {
        "agent": name,
        "avg_steps": round(mean(steps), 2),
        "med_steps": int(median(steps)),
        "avg_lines": round(mean(lines), 2),
        "avg_score": round(mean(scores), 2),
    }

def print_table(rows):
    head = f"{'Agent':<18} {'Avg Steps':>10} {'Med Steps':>10} {'Avg Lines':>10} {'Avg Score':>10}"
    print("\n" + head + "\n" + "-"*len(head))
    for r in rows:
        print(f"{r['agent']:<18} {r['avg_steps']:>10} {r['med_steps']:>10} {r['avg_lines']:>10} {r['avg_score']:>10}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--model", default="runs/ppo_tetris_shaped.zip")
    ap.add_argument("--deterministic", action="store_true")
    args = ap.parse_args()

    env_b1 = TetrisEnv()
    res_b1 = [run_episode_clears_first(env_b1) for _ in range(args.episodes)]
    env_b1.close()

    env_b2 = TetrisEnv()
    res_b2 = [run_episode_heuristic(env_b2) for _ in range(args.episodes)]
    env_b2.close()

    env_rl = TetrisEnv()
    model = PPO.load(args.model)
    res_rl = [run_episode_ppo(env_rl, model, deterministic=args.deterministic) for _ in range(args.episodes)]
    env_rl.close()

    rows = [
        summarize("Clears-First", res_b1),
        summarize("Heuristic", res_b2),
        summarize("PPO " + ("(det)" if args.deterministic else "(stoch)"), res_rl),
    ]
    print_table(rows)
