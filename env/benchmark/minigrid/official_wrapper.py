"""Wrapper to make official Farama MiniGrid envs compatible with our agent.

The official `minigrid` package uses the new gym API
(reset → (obs, info), step → (obs, reward, terminated, truncated, info)).
This wrapper bridges to the old gym API used throughout this codebase,
extracts the image array from the obs dict, and fixes each instance to a
single seed so every reset() reproduces the same layout.

Action space is kept at Discrete(7) matching our fork's action indices:
  0=left 1=right 2=forward 3=pickup 4=drop 5=toggle 6=done
Pickup/drop/toggle are harmless no-ops in maze-only environments.

NOTE: dcd_new/minigrid/ is a local fork that shadows the installed Farama
minigrid package on sys.path.  _make_official_gymnasium_env() temporarily
swaps in the site-packages version so that official MiniGrid envs can be
created without renaming the local fork.
"""

import os
import sys
import numpy as np
import gym

# -------------------------------------------------------------------------
# Helpers to create a gymnasium env from the *installed* Farama minigrid,
# bypassing the local dcd_new/minigrid/ fork that shadows it on sys.path.
# -------------------------------------------------------------------------

_official_minigrid_registered = False


def _find_site_minigrid_dir() -> str:
    """Return the site-packages directory that contains the official minigrid."""
    for p in sys.path:
        if 'site-packages' in p:
            if os.path.isdir(os.path.join(p, 'minigrid', 'envs')):
                return p
    raise ImportError(
        "Official Farama 'minigrid' not found in site-packages.\n"
        "Install with:  pip install minigrid"
    )


def _ensure_official_minigrid_registered():
    """
    Import minigrid.envs from site-packages to register all official MiniGrid
    envs with gymnasium.  Only runs once per process.

    Strategy: load the official package, keep its minigrid.envs.* submodules in
    sys.modules (so gymnasium.make() can import them later), but restore the
    local dcd_new/minigrid/ fork as the top-level 'minigrid' entry so that the
    rest of the codebase continues to see the custom MultiGrid fork.
    """
    global _official_minigrid_registered
    if _official_minigrid_registered:
        return

    sp = _find_site_minigrid_dir()

    # Stash and remove every 'minigrid*' entry so the next import picks up
    # the site-packages version instead of the local fork.
    stashed = {k: sys.modules.pop(k)
               for k in list(sys.modules)
               if k == 'minigrid' or k.startswith('minigrid.')}

    sys.path.insert(0, sp)
    try:
        import minigrid.envs  # noqa: F401 — registers all official envs with gymnasium
        # Capture every official submodule (minigrid.envs.*, minigrid.core.*, …)
        # but NOT the top-level 'minigrid' itself.
        official_submods = {k: sys.modules[k]
                            for k in list(sys.modules)
                            if k.startswith('minigrid.') and k in sys.modules}
    finally:
        if sys.path and sys.path[0] == sp:
            sys.path.pop(0)
        # Remove everything minigrid-related …
        for k in list(sys.modules):
            if k == 'minigrid' or k.startswith('minigrid.'):
                del sys.modules[k]
        # … restore the local fork as top-level 'minigrid' …
        sys.modules.update(stashed)
        # … and re-add the official submodules so gymnasium.make() can find them.
        sys.modules.update(official_submods)

    _official_minigrid_registered = True


def _make_official_gymnasium_env(env_id: str, **kwargs):
    """Create an official Farama MiniGrid gymnasium env."""
    _ensure_official_minigrid_registered()
    import gymnasium as _gymnasium
    return _gymnasium.make(env_id, **kwargs)


# -------------------------------------------------------------------------
# Wrappers
# -------------------------------------------------------------------------

class OfficialMiniGridWrapper(gym.Env):
    """Wraps an official Farama MiniGrid env to match our training env interface.

    Returns obs as np.ndarray (agent_view_size, agent_view_size, 3) uint8.
    step() returns (obs, reward, done, info) — old gym API.
    reset() reproduces the same layout via a fixed seed.
    """

    def __init__(self, env_id: str, seed: int = 0, agent_view_size: int = 5):
        kwargs = {'render_mode': 'rgb_array', 'agent_view_size': agent_view_size}
        self._env = _make_official_gymnasium_env(env_id, **kwargs)
        self._seed = seed

        # Read actual image shape from the created env (each env fixes its own view size).
        img_shape = self._env.observation_space['image'].shape  # (H, W, 3)
        self.observation_space = gym.spaces.Dict({
            'image':     gym.spaces.Box(low=0, high=255, shape=img_shape, dtype=np.uint8),
            'direction': gym.spaces.Box(low=0, high=3, shape=(1,), dtype=np.int64),
        })
        # Always Discrete(7) — use old gym type for compatibility with the rest of the codebase.
        self.action_space = gym.spaces.Discrete(7)

    # ------------------------------------------------------------------
    # gym interface (old API)
    # ------------------------------------------------------------------

    def _extract_obs(self, obs_dict) -> dict:
        return {
            'image':     obs_dict['image'],
            'direction': np.array([obs_dict['direction']], dtype=np.int64),
        }

    def reset(self, **kwargs):
        obs_dict, _ = self._env.reset(seed=self._seed)
        return self._extract_obs(obs_dict)

    def step(self, action):
        obs_dict, reward, terminated, truncated, info = self._env.step(int(action))
        done = terminated or truncated
        return self._extract_obs(obs_dict), float(reward), done, info

    def render(self, mode='rgb_array', **kwargs):
        return self._env.render()

    def close(self):
        self._env.close()

    # ------------------------------------------------------------------
    # Attributes queried by SubprocVecEnv during initialisation
    # ------------------------------------------------------------------
    spec = None
    processed_action_dim = 1  # scalar (Discrete) action


class OfficialMiniGridTrainingEnv(OfficialMiniGridWrapper):
    """Extends OfficialMiniGridWrapper with PLR / SFL training runner APIs.

    Level identity: integer seed.
      encoding        → np.array([seed], int64)   — stored in PLR buffer
      reset_random()  → random seed, returns obs
      reset_to_level(level) → fixed seed (int or np.array([seed])), returns obs
      reset_agent()   → re-reset with same seed (gym has no persistent grid)
      mutate_level()  → new random seed (simple mutation for ACCEL)
    """

    def __init__(self, env_id: str, base_seed: int = 0):
        super().__init__(env_id, seed=base_seed)
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
