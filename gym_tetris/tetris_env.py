# gym_tetris/tetris_env.py
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Dict, Any, Tuple
from engines.nuno_faria import tetris as T


def _to_array(x):
    try:
        return np.array(x)
    except Exception:
        return None


def _count_full_lines(board_2d: np.ndarray) -> int:
    if board_2d is None or board_2d.ndim != 2:
        return 0
    return int(np.sum(np.all(board_2d != 0, axis=1)))


def _count_holes(board_2d: np.ndarray) -> int:
    if board_2d is None or board_2d.ndim != 2:
        return 0
    holes = 0
    H, W = board_2d.shape
    col_nonzero = (board_2d != 0)
    for c in range(W):
        seen = False
        for r in range(H):
            if col_nonzero[r, c]:
                seen = True
            elif seen:
                holes += 1
    return int(holes)


def _obs_from(board_like) -> np.ndarray:
    b = _to_array(board_like)
    if b is None or b.ndim != 2:
        return np.zeros((20, 10), dtype=np.float32)
    return (b != 0).astype(np.float32)


class TetrisEnv(gym.Env):
    """
    Gymnasium wrapper for engines.nuno_faria.tetris.Tetris.

    Observation: (H, W) float32 occupancy (0/1)
    Action: Discrete(max_actions) -> engine key from get_next_states(),
            padded so *all* indices are valid each step.
    Reward (post-step):
        reward = +1.0 * lines_delta + 0.05 * holes_delta - 0.001
      where lines_delta = full_rows(after) - full_rows(before)
            holes_delta = holes(before) - holes(after)
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode: Optional[str] = None, max_actions: int = 64):
        super().__init__()
        self.render_mode = render_mode
        self.max_actions = int(max(1, max_actions))

        self.game = T.Tetris()

        H = len(self.game.board)
        W = len(self.game.board[0])
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(H, W), dtype=np.float32)
        self.action_space = spaces.Discrete(self.max_actions)

        # Per-step maps: local index -> engine key
        self._action_map: Dict[int, Any] = {}
        self._num_valid: int = 0  # how many real actions before padding

    # ---------- helpers ----------
    def _obs(self) -> np.ndarray:
        return _obs_from(self.game.board)

    def _enumerate_actions(self) -> int:
        """
        Build mapping for local indices [0..max_actions-1] to engine keys.
        If engine provides fewer than max_actions, we pad by repeating the first valid key,
        so any policy output index is always valid.
        """
        next_states = self.game.get_next_states()  # dict: engine_key -> state
        engine_keys = list(next_states.keys())

        k = min(len(engine_keys), self.max_actions)
        if k == 0:
            # No valid moves — keep previous map empty; caller should terminate
            self._action_map = {}
            self._num_valid = 0
            return 0

        # First fill with actual valid keys
        mapping = {i: engine_keys[i] for i in range(k)}

        # Pad up to max_actions by repeating a safe key (index 0)
        if k < self.max_actions:
            pad_key = engine_keys[0]
            for i in range(k, self.max_actions):
                mapping[i] = pad_key

        self._action_map = mapping
        self._num_valid = k
        return k

    def _normalize_engine_key(self, engine_key) -> Tuple[int, int]:
        """Return (action, rotation) for engine.play(). Handles tuple/list/dict/str/int keys."""
        try:
            if isinstance(engine_key, (tuple, list)) and len(engine_key) >= 2:
                return int(engine_key[0]), int(engine_key[1])
            if isinstance(engine_key, dict):
                act = int(engine_key.get("action", engine_key.get("x", next(iter(engine_key.values())))))
                rot = int(engine_key.get("rotation", engine_key.get("rot", 0)))
                return act, rot
            if isinstance(engine_key, str):
                s = engine_key.replace("x=", "").replace("action=", "").replace("rot=", "").replace("rotation=", "")
                parts = s.replace(" ", "").split(",")
                if len(parts) >= 2:
                    return int(parts[0]), int(parts[1])
                return int(parts[0]), 0
            if isinstance(engine_key, (int, np.integer)):
                return int(engine_key), 0
        except Exception:
            pass
        return 0, 0

    def _apply_with_autodetect(self, engine_key) -> None:
        """
        Try play(act, rot), then play(rot, act), then play(engine_key) single-arg.
        Accept the first variant that changes the board or sets game_over.
        """
        before = _to_array(self.game.board)

        # 1) play(act, rot)
        a, r = self._normalize_engine_key(engine_key)
        try:
            self.game.play(a, r)
            after = _to_array(self.game.board)
            if bool(getattr(self.game, "game_over", False)) or (after is not None and before is not None and not np.array_equal(after, before)):
                return
        except Exception:
            pass

        # 2) play(rot, act)
        try:
            self.game.play(r, a)
            after = _to_array(self.game.board)
            if bool(getattr(self.game, "game_over", False)) or (after is not None and before is not None and not np.array_equal(after, before)):
                return
        except Exception:
            pass

        # 3) play(engine_key) single-arg
        try:
            self.game.play(engine_key)
        except Exception:
            # If all variants fail, it will be a no-op; reward will be the step penalty.
            pass

    # ---------- Gymnasium API ----------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.game.reset()
        valid = self._enumerate_actions()
        return self._obs(), {"valid_actions": valid}

    def step(self, action: int):
        # Accept numpy actions and modulo-map into the available range
        try:
            import numpy as _np
            if isinstance(action, _np.ndarray):
                action = int(action.squeeze())
            else:
                action = int(action)
        except Exception:
            action = int(action)

        # If no valid actions, end episode
        if self._num_valid == 0:
            return self._obs(), -1.0, True, False, {"error": "no_valid_actions"}

        # Map action into [0, max_actions) then into [0, _num_valid) if you want only real keys
        # We padded the map, so any index is valid; still modulo for safety:
        action = action % self.action_space.n
        engine_key = self._action_map.get(action)
        if engine_key is None:
            # ultra-guard: map into real valid range
            engine_key = self._action_map[action % self._num_valid]

        # BEFORE metrics
        before_board = _to_array(self.game.board)
        lines_before = _count_full_lines(before_board)
        holes_before = _count_holes(before_board)

        # Apply
        self._apply_with_autodetect(engine_key)

        # AFTER metrics
        after_board = _to_array(self.game.board)
        lines_after = _count_full_lines(after_board)
        holes_after = _count_holes(after_board)

        lines_delta = max(0, lines_after - lines_before)
        holes_delta = max(0, holes_before - holes_after)

        # Reward: encourage line clears, reducing holes; tiny step penalty
        reward = 1.0 * lines_delta + 0.05 * holes_delta - 0.001

        terminated = bool(getattr(self.game, "game_over", False))
        truncated = False

        valid = 0
        if not terminated:
            valid = self._enumerate_actions()

        info = {
            "engine_key": engine_key,
            "valid_actions": valid,
            "lines_delta": lines_delta,
            "holes_delta": holes_delta,
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
