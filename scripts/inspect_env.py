# inspect_env.py
from gym_tetris import TetrisEnv

def pick_action(info, action_space):
    va = info.get("valid_actions", None)
    if va is None:
        return action_space.sample()
    # If it's already a single valid action id (int-like)
    if isinstance(va, (int,)):
        return int(va)
    # If it's a container of valid actions
    if isinstance(va, (list, tuple, set)):
        va_list = list(va)
        return va_list[0] if va_list else action_space.sample()
    # Unknown type — be defensive
    return action_space.sample()

env = TetrisEnv()
print("observation_space:", env.observation_space)
print("action_space:", env.action_space)

obs, info = env.reset()
print("reset info keys:", list(info.keys()))
print("type(valid_actions) on reset:", type(info.get("valid_actions")).__name__, "->", info.get("valid_actions"))

history = []
for t in range(10):
    a = pick_action(info, env.action_space)
    obs, r, term, trunc, info = env.step(a)
    history.append({
        "t": t+1,
        "a": a,
        "r": float(r),
        "linesΔ": info.get("lines_delta"),
        "valid_type": type(info.get("valid_actions")).__name__,
        "valid": info.get("valid_actions"),
        "term": term,
        "trunc": trunc
    })
    if term or trunc:
        break

print("last steps:", history[-min(5, len(history)):])
env.close()
