# gym_tetris/tetris_env.py
from __future__ import annotations

from typing import Optional, Dict, Any, Tuple, List
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# Local engine (It's NOT the pip 'gym-tetris')
from engines.nuno_faria import tetris as T


# -------------------- utilities --------------------

def _to_array(x):
    try:
        return np.array(x)
    except Exception:
        return None


def _obs_from(board_like) -> np.ndarray:
    """
    Normalize board to 0/1 float32 occupancy (H, W).
    Falls back to a 20x10 zero board if shape is unknown.
    """
    b = _to_array(board_like)
    if b is None or b.ndim != 2:
        return np.zeros((20, 10), dtype=np.float32)
    return (b != 0).astype(np.float32)


# -------------------- environment --------------------

class TetrisEnv(gym.Env):
    """
    Gymnasium wrapper for engines.nuno_faria.tetris.Tetris.

    Observation:
        Box(low=0.0, high=1.0, shape=(H, W), dtype=float32) -> binary 0/1 board occupancy

    Action:
        Discrete(max_actions).
        At each step we enumerate available placements from engine.get_next_states(),
        map them to local indices [0..max_actions-1], and pad by repeating the first key
        so any index in Discrete(max_actions) is valid.

    Reward (simple & robust):
        - Positive proportional to engine score increase this step (score_delta)
        - Tiny survival bonus per step (+0.002)

    Info:
        - "valid_actions": List[int] of currently valid local action indices
        - "action_mask": np.ndarray[bool] (True = valid) for sb3-contrib ActionMasker
        - "engine_key", "x", "rot": debug fields for chosen move
        - "score": current engine score
        - "score_delta": score increase this step (>=0)
        - "lines_delta": 1 iff score increased this step (conservative flag)
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode: Optional[str] = None, max_actions: int = 64):
        super().__init__()
        self.render_mode = render_mode
        self.max_actions = int(max(1, max_actions))
        self.game = T.Tetris()

        # Board size from engine (expected 20x10)
        H = len(self.game.board)
        W = len(self.game.board[0])
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(H, W), dtype=np.float32)
        self.action_space = spaces.Discrete(self.max_actions)

        # Action mapping/bookkeeping
        self._action_map: Dict[int, Any] = {}
        self._num_valid: int = 0
        self._valid_indices: List[int] = []
        self._action_mask: np.ndarray = np.ones(self.max_actions, dtype=bool)

        # Score tracking (engine provides get_game_score() or .score)
        self._last_score: int = 0

    # ---------- helper functions ----------

    def _obs(self) -> np.ndarray:
        return _obs_from(self.game.board)

    def _read_score(self) -> int:
        """Read engine score via method if available, else attribute."""
        if hasattr(self.game, "get_game_score") and callable(self.game.get_game_score):
            v = self.game.get_game_score()
        else:
            v = getattr(self.game, "score", 0)
        try:
            return int(v or 0)
        except Exception:
            return 0

    def _enumerate_actions(self) -> int:
        """
        Build local index -> engine key mapping from get_next_states().
        Also computes the action mask and list of valid indices.
        """
        next_states = self.game.get_next_states()
        keys = list(next_states.keys())
        k = min(len(keys), self.max_actions)

        if k == 0:
            self._action_map = {}
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

        mask = np.zeros(self.max_actions, dtype=bool)
        mask[:k] = True
        self._action_mask = mask
        return k

    @staticmethod
    def _to_x_rot(engine_key) -> Tuple[int, int]:
        """
        Convert an engine key to (x, rotation). Supports common shapes.
        """
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

    # ---------- mask API for sb3-contrib ----------

    def valid_action_mask(self) -> np.ndarray:
        """Boolean mask (True = valid) consumed by sb3-contrib ActionMasker."""
        if isinstance(self._action_mask, np.ndarray):
            return self._action_mask.copy()
        return np.ones(self.max_actions, dtype=bool)

    # ---------- Gymnasium API ----------

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.game.reset()
        self._last_score = self._read_score()
        self._enumerate_actions()
        return self._obs(), {
            "valid_actions": self._valid_indices,   # list[int]
            "action_mask": self._action_mask,       # np.bool_
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
            # End episode if no legal moves to left
            return self._obs(), 0.0, True, False, {
                "error": "no_valid_actions",
                "valid_actions": [],
                "action_mask": np.zeros(self.max_actions, dtype=bool),
                "score": int(self._last_score),
                "score_delta": 0,
                "lines_delta": 0,
            }

        action = action % self.action_space.n
        engine_key = self._action_map.get(action, self._action_map.get(0, (0, 0)))
        x, rot = self._to_x_rot(engine_key)

        # Apply move via engine
        self.game.play(x, rot)

        # Score delta -> positive signal when clears/points awarded
        score_now = self._read_score()
        score_delta = max(0, score_now - self._last_score)
        self._last_score = score_now

        # Conservative "line event" flag: 1 whenever score increased
        lines_delta = 1 if score_delta > 0 else 0

        # Reward: proportional to score gain (clipped) + tiny survival bonus
        reward = 0.1 * float(min(score_delta, 10))
        reward += 0.002

        # Termination flag from engine (supporting both attributes)
        terminated = bool(getattr(self.game, "game_over", False) or getattr(self.game, "gameover", False))
        truncated = False

        # Refresh valid actions if not terminated
        if not terminated:
            self._enumerate_actions()
        else:
            self._valid_indices = []
            self._action_mask = np.zeros(self.max_actions, dtype=bool)

        info = {
            "engine_key": engine_key,
            "x": x,
            "rot": rot,
            "valid_actions": self._valid_indices,
            "action_mask": self._action_mask,
            "score": int(score_now),
            "score_delta": int(score_delta),
            "lines_delta": int(lines_delta),
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
