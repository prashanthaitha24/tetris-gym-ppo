import numpy as np
import gymnasium as gym
from gymnasium import spaces
from engines.nuno_faria import tetris as T


class TetrisEnv(gym.Env):
    """
    Tetris environment wrapper for Stable Baselines3.

    This wraps the `nuno_faria` Tetris engine and adapts it to the
    Gym/Gymnasium API so RL agents can interact with it.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, width=10, height=20):
        super().__init__()

        self.width = width
        self.height = height

        # The engine
        self.game = T.Tetris()

        # Observation space: board height x board width with int values
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.height, self.width),
            dtype=np.uint8
        )

        # Will be updated dynamically in reset()
        self.action_space = spaces.Discrete(1)

        # Internal action mapping
        self._action_map = {}
        self._last_actions = []

    def _obs(self):
        """
        Get the current board as a numpy array.
        """
        return np.array(self.game.board, dtype=np.uint8)

    def _score_now(self):
        """
        Get the current score from the engine.
        """
        if hasattr(self.game, "score"):
            return self.game.score
        elif hasattr(self.game, "get_game_score"):
            return self.game.get_game_score()
        return 0

    def _enumerate_actions(self):
        """
        Build a mapping from discrete action indices to engine-specific actions.
        """
        states = {}
        if hasattr(self.game, "get_next_states"):
            states = self.game.get_next_states()
        elif hasattr(self.game, "get_possible_states"):
            states = self.game.get_possible_states()

        self._action_map = {i: k for i, k in enumerate(states.keys())}
        self.action_space = spaces.Discrete(len(self._action_map))
        self._last_actions = list(self._action_map.keys())
        return len(self._action_map)

    def reset(self, *, seed=None, options=None):
        """
        Reset the environment and start a new game.
        """
        super().reset(seed=seed)
        if hasattr(self.game, "reset"):
            self.game.reset()
        else:
            self.game = T.Tetris()

        valid_actions = self._enumerate_actions()
        obs = self._obs()
        return obs, {"valid_actions": valid_actions}

    def step(self, action):
        """
        Perform one step in the environment.
        """
        if action not in self._action_map:
            obs = self._obs()
            return obs, -1.0, True, False, {"error": "invalid_action_index"}

        engine_key = self._action_map[action]
        before = self._score_now()

        # --- Normalize engine_key into (act, rot) for play(act, rot) ---
        act, rot = 0, 0
        try:
            if isinstance(engine_key, (tuple, list)) and len(engine_key) >= 2:
                act, rot = int(engine_key[0]), int(engine_key[1])
            elif isinstance(engine_key, dict):
                if "action" in engine_key:
                    act = int(engine_key["action"])
                elif "x" in engine_key:
                    act = int(engine_key["x"])
                else:
                    act = int(next(iter(engine_key.values())))

                if "rotation" in engine_key:
                    rot = int(engine_key["rotation"])
                elif "rot" in engine_key:
                    rot = int(engine_key["rot"])
                else:
                    rot = 0
            elif isinstance(engine_key, (int, np.integer)):
                act, rot = int(engine_key), 0
            else:
                act, rot = 0, 0
        except Exception:
            act, rot = 0, 0

        # Call engine
        self.game.play(act, rot)

        after = self._score_now()
        reward = float(after - before)
        terminated = bool(getattr(self.game, "game_over", False))
        truncated = False

        valid = 0
        if not terminated:
            valid = self._enumerate_actions()

        info = {
            "engine_key": engine_key,
            "act": act,
            "rot": rot,
            "valid_actions": valid,
            "score": after
        }

        return self._obs(), reward, terminated, truncated, info

    def render(self, mode="human"):
        """
        Render the game.
        """
        if hasattr(self.game, "render"):
            self.game.render()

    def close(self):
        """
        Close the environment.
        """
        pass
