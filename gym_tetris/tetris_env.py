from __future__ import annotations

from typing import Optional, Dict, Any, Tuple, List
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# importing the nuno faria tetris engine
from engines.nuno_faria import tetris as T


def _to_array(x):
    try:
        return np.array(x)
    except Exception:
        return None


def _obs_from(board_like) -> np.ndarray:
    b = _to_array(board_like)
    if b is None or b.ndim != 2:
        return np.zeros((20, 10), dtype=np.float32)
    return (b != 0).astype(np.float32)


class TetrisEnv(gym.Env):
    """
    Gymnasium wrapper for engines.nuno_faria.tetris.Tetris.

    Observation: (H, W) float32 binary occupancy (stack only)
    Action:      Discrete(max_actions) over enumerated placements
    Reward:      +1 per line cleared (exact via engine hook), -10 on game over
    Info:        valid_actions, action_mask, action_keys, score, score_delta, lines_delta
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode: Optional[str] = None, max_actions: int = 64):
        super().__init__()
        self.render_mode = render_mode
        self.max_actions = int(max(1, max_actions))

        self.game = T.Tetris()
        self._hook_clear_lines()

        H = len(self.game.board)
        W = len(self.game.board[0])
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(H, W), dtype=np.float32)
        self.action_space = spaces.Discrete(self.max_actions)

        self._row_width = W
        self._action_map: Dict[int, Any] = {}
        self._action_keys: List[Any] = []
        self._num_valid: int = 0
        self._valid_indices: List[int] = []
        self._action_mask: np.ndarray = np.ones(self.max_actions, dtype=bool)

        self._last_score: int = 0
        self._last_clears_seen: int = 0

    # ---- engine hook ----
    def _hook_clear_lines(self):
        """
        Wrap the engine's _clear_lines() to record EXACT rows removed this move.
        Stores into both self._last_clears_seen and self.game.last_clears.
        """
        try:
            if hasattr(self.game, "_clear_lines"):
                original = self.game._clear_lines

                def wrapper(*args, **kwargs):
                    board_before = None
                    if args:
                        board_before = args[0]
                    elif "board" in kwargs:
                        board_before = kwargs.get("board", None)

                    cleared = 0
                    try:
                        b = _to_array(board_before)
                        if b is not None and b.ndim == 2:
                            cleared = int(np.sum(np.all(b != 0, axis=1)))
                    except Exception:
                        cleared = 0

                    result = original(*args, **kwargs)

                    try:
                        self._last_clears_seen = int(cleared)
                        setattr(self.game, "last_clears", int(cleared))
                    except Exception:
                        pass
                    return result

                self.game._clear_lines = wrapper

            if not hasattr(self.game, "last_clears"):
                setattr(self.game, "last_clears", 0)
        except Exception:
            if not hasattr(self.game, "last_clears"):
                try:
                    setattr(self.game, "last_clears", 0)
                except Exception:
                    pass

    # ---- helpers ----
    def _obs(self) -> np.ndarray:
        return _obs_from(self.game.board)

    def _read_score(self) -> int:
        if hasattr(self.game, "get_game_score") and callable(self.game.get_game_score):
            v = self.game.get_game_score()
        else:
            v = getattr(self.game, "score", 0)
        try:
            return int(v or 0)
        except Exception:
            return 0

    def _enumerate_actions(self) -> int:
        next_states = self.game.get_next_states()
        keys = list(next_states.keys())
        k = min(len(keys), self.max_actions)

        if k == 0:
            self._action_map = {}
            self._action_keys = []
            self._num_valid = 0
            self._valid_indices = []
            self._action_mask = np.zeros(self.max_actions, dtype=bool)
            return 0

        mapping = {i: keys[i] for i in range(k)}
        if k < self.max_actions:
            pad_key = keys[0]
            for i in range(k, self.max_actions):
                mapping[i] = pad_key

        self._action_map = mapping
        self._num_valid = k
        self._valid_indices = list(range(k))
        self._action_keys = [mapping[i] for i in range(self.max_actions)]

        mask = np.zeros(self.max_actions, dtype=bool)
        mask[:k] = True
        self._action_mask = mask
        return k

    @staticmethod
    def _to_x_rot(engine_key) -> Tuple[int, int]:
        if isinstance(engine_key, (tuple, list)) and len(engine_key) >= 2:
            return int(engine_key[0]), int(engine_key[1])
        if isinstance(engine_key, dict):
            x = int(engine_key.get("x", engine_key.get("action", 0)))
            rot = int(engine_key.get("rotation", engine_key.get("rot", 0)))
            return x, rot
        if isinstance(engine_key, str):
            s = engine_key.replace("x=", "").replace("action=", "").replace("rot=", "").replace("rotation=", "")
            parts = s.replace(" ", "").split(",")
            if len(parts) >= 2:
                return int(parts[0]), int(parts[1])
            return int(parts[0]), 0
        if isinstance(engine_key, (int, np.integer)):
            return int(engine_key), 0
        return 0, 0

    # ---- sb3-contrib mask API ----
    def valid_action_mask(self) -> np.ndarray:
        return self._action_mask.copy() if isinstance(self._action_mask, np.ndarray) \
               else np.ones(self.max_actions, dtype=bool)

    # ---- Gymnasium API ----
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.game.reset()
        self._last_clears_seen = 0
        try:
            setattr(self.game, "last_clears", 0)
        except Exception:
            pass

        self._last_score = self._read_score()
        self._enumerate_actions()
        return self._obs(), {
            "valid_actions": self._valid_indices,
            "action_mask": self._action_mask,
            "action_keys": self._action_keys,
        }

    def step(self, action: int):
        try:
            import numpy as _np
            if isinstance(action, _np.ndarray):
                action = int(action.squeeze())
            else:
                action = int(action)
        except Exception:
            action = int(action)

        if self._num_valid == 0:
            info = {
                "error": "no_valid_actions",
                "valid_actions": [],
                "action_mask": np.zeros(self.max_actions, dtype=bool),
                "action_keys": self._action_keys,
                "score": int(self._last_score),
                "score_delta": 0,
                "lines_delta": 0,
            }
            return self._obs(), -10.0, True, False, info

        action = action % self.action_space.n
        engine_key = self._action_map.get(action, self._action_map.get(0, (0, 0)))
        x, rot = self._to_x_rot(engine_key)

        # BEFORE
        score_before = self._read_score()

        # Reset per move clear counter to avoid stale values
        self._last_clears_seen = 0
        try:
            setattr(self.game, "last_clears", 0)
        except Exception:
            pass

        # Engine will call _clear_lines internally if needed
        self.game.play(x, rot)

        score_now = self._read_score()
        score_delta = max(0, score_now - score_before)

        # Clears captured by the hook
        lines_cleared = int(getattr(self.game, "last_clears", self._last_clears_seen) or 0)
        lines_cleared = max(0, lines_cleared)

        terminated = bool(getattr(self.game, "game_over", False) or getattr(self.game, "gameover", False))

        reward = float(lines_cleared)
        if terminated:
            reward -= 10.0

        truncated = False
        self._last_score = score_now

        if not terminated:
            self._enumerate_actions()
        else:
            self._valid_indices = []
            self._action_mask = np.zeros(self.max_actions, dtype=bool)

        info = {
            "engine_key": engine_key,
            "x": x, "rot": rot,
            "valid_actions": self._valid_indices,
            "action_mask": self._action_mask,
            "action_keys": self._action_keys,
            "score": int(score_now),
            "score_delta": int(score_delta),
            "lines_delta": int(lines_cleared),  
        }
        return self._obs(), float(reward), terminated, truncated, info

    def render(self):
        if hasattr(self.game, "render") and callable(self.game.render):
            try:
                return self.game.render()
            except Exception:
                pass

    def close(self):
        pass
