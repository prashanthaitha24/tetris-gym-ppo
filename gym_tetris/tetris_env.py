import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Dict, Any

class _EngineAdapter:
    def __init__(self, width=10, height=20):
        try:
            from engines.nuno_faria.tetris import Tetris  # type: ignore
        except Exception as e:
            raise ImportError("Missing engine. Run `python3.13 prepare_engine.py`.") from e
        self._T = Tetris
        self.game = Tetris(width=width, height=height)

    def reset(self):
        self.game = self._T(width=getattr(self.game, 'width', 10), height=getattr(self.game, 'height', 20))
        return self.game

    def enumerate_actions(self):
        if hasattr(self.game, 'get_possible_states'):
            out = {}
            for i, (k, v) in enumerate(self.game.get_possible_states().items()):
                if isinstance(v, dict):
                    board = v.get('board')
                    lines = v.get('lines', 0)
                else:
                    board, lines = v[0], (v[1] if len(v) > 1 else 0)
                out[i] = {'board': board, 'lines': int(lines)}
            return out
        return {0: {'board': self.game.board.copy(), 'lines': 0}}

    def apply_action(self, action_id: int):
        if hasattr(self.game, 'apply_action'):
            return self.game.apply_action(int(action_id))
        self.game.game_over = True
        return 0.0, True, {}

    def get_board(self):
        return self.game.board

    def score(self):
        return getattr(self.game, 'score', 0)

class TetrisEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, width: int = 10, height: int = 20, render_mode: Optional[str] = None):
        super().__init__()
        self.width = width
        self.height = height
        self.render_mode = render_mode
        self.adapter = _EngineAdapter(width, height)

        self.observation_space = spaces.Box(0.0, 1.0, shape=(height, width), dtype=np.float32)
        self.max_actions = 64
        self.action_space = spaces.Discrete(self.max_actions)
        self._last_actions: Dict[int, Dict[str, Any]] = {}

    def _enumerate_actions(self):
        m = self.adapter.enumerate_actions()
        keys = sorted(m.keys())[: self.max_actions]
        self._last_actions = {i: m[k] for i, k in enumerate(keys)}
        return self._last_actions

    def _obs(self):
        return (self.adapter.get_board() > 0).astype(np.float32)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.adapter.reset()
        self._enumerate_actions()
        return self._obs(), {"valid_actions": len(self._last_actions), "score": self.adapter.score()}

    def step(self, action: int):
        if action not in self._last_actions:
            return self._obs(), -1.0, True, False, {"err": "bad_action"}
        r, term, info = self.adapter.apply_action(int(action))
        if not term:
            self._enumerate_actions()
        return self._obs(), float(r), bool(term), False, {"valid_actions": len(self._last_actions), "score": self.adapter.score()}
