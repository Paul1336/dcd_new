"""MinGrid VLM-generated task environment.

Loads pre-generated grid encodings (numpy arrays) from disk and exposes them
as a fixed task pool, analogous to IphyreAdversarialVLM4kEnv for iphyre.

Registered env:
    MultiGrid-AdversarialVLM-v0
"""

import os
import random

import numpy as np
from networkx import grid_graph

from minigrid import adversarial, register
from minigrid.adversarial import AdversarialEnv


def load_vlm_gen_tasks(task_dir_list: list) -> list:
    """Load encoding.npy files from task_NNNNNN subdirs in each directory.

    Returns a list of numpy arrays, each of shape (size, size, 3).
    """
    encodings = []
    for task_dir in task_dir_list:
        if not os.path.isdir(task_dir):
            print(f'[vlm_adversarial] Warning: directory not found: {task_dir}')
            continue
        for name in sorted(os.listdir(task_dir)):
            d = os.path.join(task_dir, name)
            enc_path = os.path.join(d, 'encoding.npy')
            if os.path.isfile(enc_path):
                encodings.append(np.load(enc_path))
    print(f'[vlm_adversarial] Loaded {len(encodings)} encodings '
          f'from {len(task_dir_list)} director{"ies" if len(task_dir_list) != 1 else "y"}')
    return encodings


class MultiGridAdversarialVLMEnv(AdversarialEnv):
    """AdversarialEnv with a fixed pool of VLM-generated grid encodings.

    Each reset() samples a random encoding from the pool and restores the
    grid directly via reset_to_encoding(), bypassing the adversary.

    Exposes subsampled_env_ids (list of numpy arrays) so LearnabilitySampler
    can access the pool via remote_attr('subsampled_env_ids').
    """

    def __init__(self, size=15, seed=0, task_dir_list=None):
        super().__init__(
            n_clutter=60,
            size=size,
            choose_goal_last=True,
            resample_n_clutter=True,
            seed=seed,
            max_steps=250,
        )
        if task_dir_list is None:
            from env.benchmark.minigrid.datapath import vlm_task_dir_list
            task_dir_list = vlm_task_dir_list

        encodings = load_vlm_gen_tasks(task_dir_list)
        if not encodings:
            raise RuntimeError(
                f'No VLM tasks found in: {task_dir_list}\n'
                'Generate tasks first:\n'
                '  python minigrid_data/generate_tasks.py --num-tasks 100'
            )
        self.subsampled_env_ids = encodings
        print(f'MultiGridAdversarialVLMEnv: {len(self.subsampled_env_ids)} tasks '
              f'(grid {size}x{size})')

    def reset(self):
        enc = random.choice(self.subsampled_env_ids)
        return self.reset_to_level(enc)

    def reset_to_level(self, level):
        """Restore grid from a numpy encoding (bypasses adversary steps).

        Initialises grid structure, resets graph/wall_locs to empty state,
        then applies the encoding directly.
        """
        self._gen_grid(self.width, self.height)
        self.step_count = 0
        # Reset adversary bookkeeping to a clean state before applying encoding
        self.wall_locs = []
        self.graph = grid_graph(dim=[self.width - 2, self.height - 2])
        return self.reset_to_encoding(level)


if hasattr(__loader__, 'name'):
    _module_path = __loader__.name
elif hasattr(__loader__, 'fullname'):
    _module_path = __loader__.fullname

register.register(
    env_id='MultiGrid-AdversarialVLM-v0',
    entry_point=_module_path + ':MultiGridAdversarialVLMEnv',
    max_episode_steps=250,
)
