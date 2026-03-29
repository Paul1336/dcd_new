"""Wrapper to make official Farama MiniGrid envs compatible with our agent.

The official `minigrid` package uses the new gym API
(reset → (obs, info), step → (obs, reward, terminated, truncated, info)).
This wrapper bridges to the old gym API used throughout this codebase,
extracts the image array from the obs dict, and fixes each instance to a
single seed so every reset() reproduces the same layout.

Action space is kept at Discrete(7) matching our fork's action indices:
  0=left 1=right 2=forward 3=pickup 4=drop 5=toggle 6=done
Pickup/drop/toggle are harmless no-ops in maze-only environments.
"""

import numpy as np
import gym


class OfficialMiniGridWrapper(gym.Env):
    """Wraps an official Farama MiniGrid env to match our training env interface.

    Returns obs as np.ndarray (agent_view_size, agent_view_size, 3) uint8.
    step() returns (obs, reward, done, info) — old gym API.
    reset() reproduces the same layout via a fixed seed.
    """

    def __init__(self, env_id: str, seed: int = 0,
                 agent_view_size: int = 5, see_through_walls: bool = True):
        try:
            import minigrid  # noqa: F401 — triggers gymnasium registration of all MiniGrid envs
        except ImportError:
            raise ImportError(
                "Official 'minigrid' (Farama) package not found.\n"
                "Install with:  pip install minigrid"
            )
        import gymnasium as _gymnasium
        self._env = _gymnasium.make(
            env_id,
            agent_view_size=agent_view_size,
            see_through_walls=see_through_walls,
            render_mode=None,
        )
        self._seed = seed

        self.observation_space = gym.spaces.Box(
            low=0, high=255,
            shape=(agent_view_size, agent_view_size, 3),
            dtype=np.uint8,
        )
        # Always Discrete(7) — use old gym type for compatibility with the rest of the codebase.
        self.action_space = gym.spaces.Discrete(7)

    # ------------------------------------------------------------------
    # gym interface (old API)
    # ------------------------------------------------------------------

    def reset(self, **kwargs):
        obs_dict, _ = self._env.reset(seed=self._seed)
        return obs_dict['image']

    def step(self, action):
        obs_dict, reward, terminated, truncated, info = self._env.step(int(action))
        done = terminated or truncated
        return obs_dict['image'], float(reward), done, info

    def close(self):
        self._env.close()

    # ------------------------------------------------------------------
    # Spec (required by SubprocVecEnv.get_spaces_spec)
    # ------------------------------------------------------------------
    spec = None


class OfficialMiniGridTrainingEnv(OfficialMiniGridWrapper):
    """Extends OfficialMiniGridWrapper with PLR / SFL training runner APIs.

    Level identity: integer seed.
      encoding        → np.array([seed], int64)   — stored in PLR buffer
      reset_random()  → random seed, returns obs
      reset_to_level(level) → fixed seed (int or np.array([seed])), returns obs
      reset_agent()   → re-reset with same seed (gym has no persistent grid)
      mutate_level()  → new random seed (simple mutation for ACCEL)
    """

    def __init__(self, env_id: str, base_seed: int = 0,
                 agent_view_size: int = 5, see_through_walls: bool = True):
        super().__init__(env_id, seed=base_seed,
                         agent_view_size=agent_view_size,
                         see_through_walls=see_through_walls)
        self._current_seed = base_seed
        self._np_random = np.random.RandomState(base_seed)

    @property
    def encoding(self) -> np.ndarray:
        return np.array([self._current_seed], dtype=np.int64)

    def _set_seed(self, seed: int):
        self._current_seed = int(seed)
        self._seed = self._current_seed

    def reset_random(self):
        self._set_seed(int(self._np_random.randint(0, 2 ** 31)))
        return self.reset()

    def reset_agent(self):
        return self.reset()

    def reset_to_level(self, level):
        if isinstance(level, np.ndarray):
            level = int(level.flat[0])
        self._set_seed(int(level))
        return self.reset()

    def mutate_level(self, num_edits=1):
        self._set_seed(int(self._np_random.randint(0, 2 ** 31)))
        return self.reset()
