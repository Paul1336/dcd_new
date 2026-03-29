"""MinGrid test suite definitions.

Each suite returns (env_names, env_fns):
  env_names: List[str]        — unique identifier per task (used as dict key in results)
  env_fns:   List[Callable]   — zero-arg thunks, each returns a gym-compatible env

Supported suite names
---------------------
  MultiGrid-RandomGenerated-v0   : procedurally generated, fixed seeds
  MultiGrid-VLMSampled-v0        : sampled from VLM-generated pool on disk
  MultiGrid-FourRooms-v0         : official MiniGrid FourRooms (fixed seeds)
  MultiGrid-SimpleCrossing-v0    : official MiniGrid SimpleCrossing (fixed seeds)
  MultiGrid-Maze-v0              : official MiniGrid Maze (fixed seeds)
"""

import random
from typing import List, Callable, Tuple

import numpy as np

from env.registration import make as gym_make
from env.benchmark.minigrid.datapath import vlm_task_dir_list
from env.benchmark.minigrid.vlm_adversarial import load_vlm_gen_tasks

# -----------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------

MINIGRID_SUITE_NAMES = frozenset({
    'MultiGrid-RandomGenerated-v0',
    'MultiGrid-VLMSampled-v0',
    'MultiGrid-FourRooms-v0',
    'MultiGrid-SimpleCrossing-v0',
    'MultiGrid-Maze-v0',
})

_SUITE_NUM_TASKS = 20
_BASE_SEED = 0

_OFFICIAL_ENV_IDS = {
    'MultiGrid-FourRooms-v0':       ('MiniGrid-FourRooms-v0',          'fourrooms'),
    'MultiGrid-SimpleCrossing-v0':  ('MiniGrid-SimpleCrossingS11N5-v0', 'simplecrossing'),
    'MultiGrid-Maze-v0':            ('MiniGrid-MultiRoom-N6-v0',             'maze'),
}

# -----------------------------------------------------------------------
# Internal fixed-level wrapper
# -----------------------------------------------------------------------

class _FixedLevelEnv:
    """Thin wrapper: every reset() restores the same grid encoding.

    Compatible with SubprocVecEnv (duck-typed gym.Env).
    obs returned as np.ndarray image (agent_view_size, agent_view_size, 3).
    """

    def __init__(self, env, encoding: np.ndarray):
        self._env = env
        self._encoding = encoding
        self.observation_space = env.observation_space  # keep full Dict space
        self.action_space = env.action_space
        self.spec = None

    def reset(self, **kwargs):
        obs = self._env.reset_to_level(self._encoding)
        # gym.make wraps envs in TimeLimit, which requires _elapsed_steps != None
        # before step() can be called.  reset_to_level() bypasses TimeLimit.reset(),
        # so we initialise the counter here.
        env = self._env
        while env is not None:
            if hasattr(env, '_elapsed_steps'):
                env._elapsed_steps = 0
            env = getattr(env, 'env', None)
        return obs

    def step(self, action):
        return self._env.step(action)  # full dict obs

    def close(self):
        self._env.close()


# -----------------------------------------------------------------------
# Suite builders
# -----------------------------------------------------------------------

def _make_random_suite(
    num_tasks: int = _SUITE_NUM_TASKS,
    base_seed: int = _BASE_SEED,
) -> Tuple[List[str], List[Callable]]:
    """Generate num_tasks random levels from GoalLastAdversarialEnv with fixed seed."""
    env = gym_make('MultiGrid-GoalLastAdversarial-v0')
    env.seed(base_seed)

    encodings = []
    for _ in range(num_tasks):
        env.reset_random()
        encodings.append(env.encoding.copy())
    env.close()

    env_names = [f'random_{i:04d}' for i in range(num_tasks)]
    env_fns = [
        (lambda enc: lambda: _FixedLevelEnv(
            gym_make('MultiGrid-GoalLastAdversarial-v0'), enc
        ))(e)
        for e in encodings
    ]
    return env_names, env_fns


def _make_vlm_suite(
    num_tasks: int = _SUITE_NUM_TASKS,
    base_seed: int = _BASE_SEED,
) -> Tuple[List[str], List[Callable]]:
    """Sample up to num_tasks encodings from the on-disk VLM task pool."""
    all_encodings = load_vlm_gen_tasks(vlm_task_dir_list)
    if not all_encodings:
        raise RuntimeError(
            'No VLM tasks found. Generate tasks first:\n'
            '  python minigrid_data/generate_tasks.py --num-tasks 1000'
        )
    rng = random.Random(base_seed)
    if len(all_encodings) > num_tasks:
        encodings = rng.sample(all_encodings, num_tasks)
    else:
        encodings = list(all_encodings)

    env_names = [f'vlm_{i:04d}' for i in range(len(encodings))]
    env_fns = [
        (lambda enc: lambda: _FixedLevelEnv(
            gym_make('MultiGrid-GoalLastAdversarial-v0'), enc
        ))(e)
        for e in encodings
    ]
    return env_names, env_fns


def _make_official_suite(
    suite_name: str,
    num_tasks: int = _SUITE_NUM_TASKS,
    base_seed: int = _BASE_SEED,
) -> Tuple[List[str], List[Callable]]:
    """Wrap official Farama MiniGrid envs, one seed per task."""
    from env.benchmark.minigrid.official_wrapper import OfficialMiniGridWrapper

    official_env_id, short = _OFFICIAL_ENV_IDS[suite_name]
    env_names = [f'{short}_seed{i:04d}' for i in range(num_tasks)]
    env_fns = [
        (lambda eid, s: lambda: OfficialMiniGridWrapper(eid, seed=s))(
            official_env_id, base_seed + i
        )
        for i in range(num_tasks)
    ]
    return env_names, env_fns


# -----------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------

def load_minigrid_test_suite(
    suite_name: str,
    num_tasks: int = _SUITE_NUM_TASKS,
) -> Tuple[List[str], List[Callable]]:
    """Return (env_names, env_fns) for the named MinGrid test suite."""
    if suite_name not in MINIGRID_SUITE_NAMES:
        raise ValueError(
            f'Unknown MinGrid suite: {suite_name!r}\n'
            f'Available: {sorted(MINIGRID_SUITE_NAMES)}'
        )
    if suite_name == 'MultiGrid-RandomGenerated-v0':
        return _make_random_suite(num_tasks)
    if suite_name == 'MultiGrid-VLMSampled-v0':
        return _make_vlm_suite(num_tasks)
    return _make_official_suite(suite_name, num_tasks)
