# scripts/check_valid_actions.py
from gym_tetris import TetrisEnv
from collections import Counter

env = TetrisEnv()
obs, info = env.reset()
counts = []
for _ in range(200):
    k = len(info.get("valid_actions", []))
    counts.append(k)
    a = (info.get("valid_actions") or [0])[0]
    obs, r, term, trunc, info = env.step(a)
    if term or trunc:
        obs, info = env.reset()
print("avg_k:", sum(counts)/len(counts), "min_k:", min(counts), "max_k:", max(counts), Counter(counts).most_common(5))
env.close()
