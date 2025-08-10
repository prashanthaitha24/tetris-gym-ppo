import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Dict, Any, Tuple


"""
Tetris Gymnasium Environment Wrapper (for nuno-faria/tetris-ai engine)

This wrapper exposes a Gymnasium-compatible API around the engine so you can
use Stable-Baselines3 PPO out of the box.

It is intentionally defensive because forks of the engine differ a bit:
- Some constructors accept (width, height); others do not.
- Some expose get_possible_states()/apply_action(); others do not.

If a method is missing, we fall back gracefully and end the episode rather
than crashing, so you can iterate quickly.
"""


# -----------------------------
# Engine Adapter
# -----------------------------
class _EngineAdapter:
    """
    Thin adapter around engines.nuno_faria.tetris.Tetris to normalize calls.
    """

    def __init__(self, width: int = 10, height: int = 20):
        try:
            from engines.nuno_faria.tetris import Tetris  # type: ignore
        except Exception as e:
            raise ImportError(
                "Could not import engines.nuno_faria.tetris.Tetris. "
                "Run `python prepare_engine.py` first (and ensure dependencies like opencv-python-headless are installed)."
            ) from e

        self._T = Tetris

        # Try engines that accept (width, height). If not, fall back to no-arg.
        try:
            self.game = Tetris(width=width, height=height)
        except TypeError:
            self.game = Tetris()

        # Cache dimensions if the engine exposes them; otherwise use defaults.
        self.width = getattr(self.game, "width", width)
        self.height = getattr(self.game, "height", height)

    def reset(self):
        """Recreate a fresh engine instance each episode."""
        try:
            self.game = self._T(width=self.width, height=self.height)
        except TypeError:
            self.game = self._T()
        return self.game

    # --------- Capability checks & adapters ---------
    def has_enumeration(self) -> bool:
        return hasattr(self.game, "get_possible_states")

    def has_apply(self) -> bool:
        return hasattr(self.game, "apply_action")

    def enumerate_actions(self) -> Dict[int, Dict[str, Any]]:
        """
        Return a map: local_action_index -> { 'board': np.ndarray, 'lines': int, ... }
        If the engine provides get_possible_states(), we adapt its return.
        Otherwise, we return a single "no-op" action to keep training flow alive.
        """
        if self.has_enumeration():
            states = self.game.get_possible_states()  # type: ignore[attr-defined]
            out: Dict[int, Dict[str, Any]] = {}
            # states may be {engine_action_id -> (board, lines, ...)} or dicts
            for i, (k, v) in enumerate(states.items()):
                if isinstance(v, dict):
                    board = v.get("board")
                    lines = int(v.get("lines", 0))
                    meta = v
                else:
                    # assume tuple-like
                    board = v[0]
                    lines = int(v[1]) if len(v) > 1 else 0
                    meta = {"raw": v}
                out[i] = {"board": board, "lines": lines, "meta": meta, "engine_key": k}
            return out

        # Fallback: single action that effectively terminates when applied
        return {0: {"board": self.game.board.copy(), "lines": 0, "meta": {}, "engine_key": 0}}

    def apply_action(self, local_index: int, last_actions: Dict[int, Dict[str, Any]]) -> Tuple[float, bool, Dict[str, Any]]:
        """
        Apply the chosen action. If the engine supports apply_action, forward it.
        We pass the *local* index by default; if your engine requires its own
        action key, you can map via last_actions[local_index]['engine_key'].
        """
        if self.has_apply():
            # Prefer passing the engine's real key if available
            engine_key = last_actions.get(local_index, {}).get("engine_key", local_index)
            return self.game.apply_action(int(engine_key))  # type: ignore[attr-defined]

        # Fallback: no apply path — end the episode with small penalty
        setattr(self.game, "game_over", True)
        return -1.0, True, {"reason": "apply_action not available in engine"}

    def get_board(self) -> np.ndarray:
        return np.asarray(self.game.board)

    def score(self) -> int:
        return int(getattr(self.game, "score", 0))

    def game_over(self) -> bool:
        return bool(getattr(self.game, "game_over", False))

    def render_rgb(self) -> Optional[np.ndarray]:
        if hasattr(self.game, "render"):
            try:
                return self.game.render(mode="rgb_array")  # type: ignore[attr-defined]
            except TypeError:
                # Some engines expose render() without mode
                arr = self.game.render()  # type: ignore[attr-defined]
                return np.asarray(arr) if arr is not None else None
        return None


# -----------------------------
# Gymnasium Env
# -----------------------------
class TetrisEnv(gym.Env):
    """
    Observation: (H, W) float32 grid of 0/1 occupancy.
    Action: Discrete(N) enumerated placements (rotations × columns). N is capped.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, width: int = 10, height: int = 20, render_mode: Optional[str] = None, max_actions: int = 64):
        super().__init__()
        self.render_mode = render_mode
        self.adapter = _EngineAdapter(width=width, height=height)

        # Use adapter-reported dims to size the observation space
        H = getattr(self.adapter, "height", height)
        W = getattr(self.adapter, "width", width)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(H, W), dtype=np.float32)

        # Discrete action space (we cap to keep a fixed space for SB3)
        self.max_actions = int(max(1, max_actions))
        self.action_space = spaces.Discrete(self.max_actions)

        # Cache of actions enumerated for the *current* piece
        self._last_actions: Dict[int, Dict[str, Any]] = {}

    # --------- Helper methods ---------
    def _enumerate_actions(self) -> Dict[int, Dict[str, Any]]:
        mapping = self.adapter.enumerate_actions()
        # Enforce a fixed upper bound
        keys = list(mapping.keys())[: self.max_actions]
        limited = {i: mapping[k] for i, k in enumerate(keys)}
        self._last_actions = limited
        return limited

    def _obs(self) -> np.ndarray:
        board = (self.adapter.get_board() > 0).astype(np.float32)
        return board

    # --------- Gymnasium API ---------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.adapter.reset()
        self._enumerate_actions()
        obs = self._obs()
        info = {"valid_actions": len(self._last_actions), "score": self.adapter.score()}
        return obs, info

    def step(self, action: int):
        # Validate action index against current enumeration
        if action not in self._last_actions:
            # End episode to avoid undefined behavior
            obs = self._obs()
            return obs, -1.0, True, False, {"error": "invalid_action_index"}

        reward, terminated, info = self.adapter.apply_action(int(action), self._last_actions)

        # If still alive, enumerate actions for the next piece
        if not terminated and not self.adapter.game_over():
            self._enumerate_actions()

        obs = self._obs()
        done = bool(terminated or self.adapter.game_over())
        info = {**(info or {}), "valid_actions": len(self._last_actions), "score": self.adapter.score()}
        return obs, float(reward), done, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self.adapter.render_rgb()
        # For "human" you could add a simple print or pygame viewer if desired.

    def close(self):
        pass
