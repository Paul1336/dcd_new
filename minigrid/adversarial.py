# coding=utf-8
# Copyright 2021 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""An environment which is built by a learning adversary."""
import random
import gym
import gym_minigrid.minigrid as minigrid
import networkx as nx
from networkx import grid_graph
import numpy as np

from . import multigrid
from . import register


EDITOR_ACTION_SPACES = {
    'walls_none': {0: '-', 1: '.'},
    'walls_none_goal': {0: '-', 1: '.', 2: 'g'},
    'walls_none_agent_goal': {0: '-', 1: '.', 2: 'a', 3: 'g'},
}


class AdversarialEnv(multigrid.MultiGridEnv):
    """Grid world where an adversary builds the environment the agent plays."""

    def __init__(self, n_clutter=50, resample_n_clutter=False, size=15,
                 agent_view_size=5, max_steps=250, goal_noise=0.,
                 random_z_dim=50, choose_goal_last=False, see_through_walls=True,
                 seed=0, editor_actions='walls_none_agent_goal',
                 fixed_environment=False):
        self.agent_start_pos = None
        self.goal_pos = None
        self.n_clutter = n_clutter
        self.resample_n_clutter = resample_n_clutter
        self.goal_noise = goal_noise
        self.random_z_dim = random_z_dim
        self.choose_goal_last = choose_goal_last
        self.n_clutter_sampled = False
        self.adversary_max_steps = self.n_clutter + 2

        super().__init__(
            n_agents=1,
            minigrid_mode=True,
            grid_size=size,
            max_steps=max_steps,
            agent_view_size=agent_view_size,
            see_through_walls=see_through_walls,
            competitive=True,
            seed=seed,
            fixed_environment=fixed_environment,
        )

        self.reset_metrics()
        self.editor_actions = list(EDITOR_ACTION_SPACES[editor_actions].values())

        self.adversary_action_dim = (size - 2) ** 2
        self.adversary_action_space = gym.spaces.Discrete(self.adversary_action_dim)
        self.adversary_ts_obs_space = gym.spaces.Box(
            low=0, high=self.adversary_max_steps, shape=(1,), dtype='uint8')
        self.adversary_randomz_obs_space = gym.spaces.Box(
            low=0, high=1.0, shape=(random_z_dim,), dtype=np.float32)
        self.adversary_image_obs_space = gym.spaces.Box(
            low=0, high=255, shape=(self.width, self.height, 3), dtype='uint8')
        self.adversary_observation_space = gym.spaces.Dict({
            'image': self.adversary_image_obs_space,
            'time_step': self.adversary_ts_obs_space,
            'random_z': self.adversary_randomz_obs_space,
        })

        self.graph = grid_graph(dim=[size - 2, size - 2])
        self.wall_locs = []

    def _resample_n_clutter(self):
        n_clutter = np.random.randint(0, self.n_clutter)
        self.adversary_max_steps = n_clutter + 2
        self.n_clutter_sampled = True
        return n_clutter

    @property
    def processed_action_dim(self):
        return 1

    @property
    def encoding(self):
        return self.grid.encode()

    def _gen_grid(self, width, height):
        self.grid = multigrid.Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

    def get_goal_x(self):
        return -1 if self.goal_pos is None else self.goal_pos[0]

    def get_goal_y(self):
        return -1 if self.goal_pos is None else self.goal_pos[1]

    def reset_metrics(self):
        self.distance_to_goal = -1
        self.n_clutter_placed = 0
        self.passable = -1
        self.shortest_path_length = (self.width - 2) * (self.height - 2) + 1

    def compute_metrics(self):
        self.n_clutter_placed = self._count_walls()
        self.compute_shortest_path()

    def reset(self):
        self.graph = grid_graph(dim=[self.width - 2, self.height - 2])
        self.wall_locs = []
        self.step_count = 0
        self.adversary_step_count = 0

        if self.resample_n_clutter:
            self.n_clutter_sampled = False

        self.agent_start_dir = self._rand_int(0, 4)
        self.reset_agent_status()
        self.agent_start_pos = None
        self.goal_pos = None
        self.done = False
        self.reset_metrics()
        self._gen_grid(self.width, self.height)

        image = self.grid.encode()
        obs = {
            'image': image,
            'time_step': [self.adversary_step_count],
            'random_z': self.generate_random_z(),
        }
        return obs

    def reset_agent_status(self):
        self.agent_pos = [None] * self.n_agents
        self.agent_dir = [self.agent_start_dir] * self.n_agents
        self.done = [False] * self.n_agents
        self.carrying = [None] * self.n_agents

    def reset_agent(self):
        for a in range(self.n_agents):
            if self.agent_pos[a] is not None:
                self.grid.set(self.agent_pos[a][0], self.agent_pos[a][1], None)

        self.reset_agent_status()

        if self.agent_start_pos is None:
            raise ValueError('Trying to place agent at empty start position.')
        else:
            self.place_agent_at_pos(0, self.agent_start_pos, rand_dir=False)

        for a in range(self.n_agents):
            assert self.agent_pos[a] is not None
            assert self.agent_dir[a] is not None
            start_cell = self.grid.get(*self.agent_pos[a])
            if not (start_cell.type == 'agent' or
                    start_cell is None or start_cell.can_overlap()):
                raise ValueError('Wrong object in agent start position.')

        self.step_count = 0
        obs = self.gen_obs()
        return obs

    def reset_to_level(self, level):
        self.reset()
        if isinstance(level, str):
            actions = [int(a) for a in level.split()]
            if self.resample_n_clutter:
                self.adversary_max_steps = len(actions)
            for a in actions:
                obs, _, done, _ = self.step_adversary(a)
                if done:
                    obs = self.reset_agent()
        else:
            obs = self.reset_to_encoding(level)
        return obs

    def reset_to_encoding(self, encoding):
        self.grid.set_encoding(encoding, multigrid_env=self)
        self.compute_metrics()
        return self.reset_agent()

    def _clean_loc(self, x, y):
        self.remove_wall(x, y)
        if isinstance(self.grid.get(x, y), minigrid.Goal):
            self.goal_pos = None
        elif isinstance(self.grid.get(x, y), multigrid.Agent):
            self.agent_start_pos = None
        self.grid.set(x, y, None)

    def _free_xy_from_mask(self, free_mask):
        free_idx = free_mask.flatten().nonzero()[0]
        free_loc = np.random.choice(free_idx)
        mask_w, mask_h = free_mask.shape
        x = free_loc % mask_w
        y = free_loc // mask_w
        return x, y

    def mutate_level(self, num_edits=1):
        num_tiles = (self.width - 2) * (self.height - 2)
        edit_locs = list(set(np.random.randint(0, num_tiles, num_edits)))
        action_idx = np.random.randint(0, len(self.editor_actions), len(edit_locs))
        actions = [self.editor_actions[i] for i in action_idx]

        free_mask = ~self.wall_mask
        free_mask[self.agent_start_pos[1] - 1, self.agent_start_pos[0] - 1] = False
        free_mask[self.goal_pos[1] - 1, self.goal_pos[0] - 1] = False

        for loc, a in zip(edit_locs, actions):
            x = loc % (self.width - 2) + 1
            y = loc // (self.width - 2) + 1
            self._clean_loc(x, y)

            if a == '-':
                self.put_obj(minigrid.Wall(), x, y)
                self.wall_locs.append((x - 1, y - 1))
                self.n_clutter_placed += 1
                free_mask[y - 1, x - 1] = False
            elif a == '.':
                self.remove_wall(x, y)
                self.grid.set(x, y, None)
                free_mask[y - 1, x - 1] = True
            elif a == 'a':
                if self.agent_start_pos is not None:
                    ax, ay = self.agent_start_pos
                    self.grid.set(ax, ay, None)
                    free_mask[ay - 1, ax - 1] = True
                self.place_one_agent(0, top=(x, y), size=(1, 1))
                self.agent_start_pos = np.array((x, y))
                free_mask[y - 1, x - 1] = False
            elif a == 'g':
                if self.goal_pos is not None:
                    gx, gy = self.goal_pos
                    self.grid.set(gx, gy, None)
                    free_mask[gy - 1, gx - 1] = True
                self.put_obj(minigrid.Goal(), x, y)
                self.goal_pos = np.array((x, y))
                free_mask[y - 1, x - 1] = False

        if self.goal_pos is None:
            x, y = self._free_xy_from_mask(free_mask)
            free_mask[y, x] = False
            x += 1; y += 1
            self.put_obj(minigrid.Goal(), x, y)
            self.goal_pos = np.array((x, y))

        if self.agent_start_pos is None:
            x, y = self._free_xy_from_mask(free_mask)
            free_mask[y, x] = False
            x += 1; y += 1
            self.place_one_agent(0, top=(x, y), size=(1, 1))
            self.agent_start_pos = np.array((x, y))

        self.graph = grid_graph(dim=[self.width - 2, self.height - 2])
        self.step_count = 0
        self.adversary_step_count = 0
        self.reset_metrics()
        self.compute_metrics()
        return self.reset_agent()

    def remove_wall(self, x, y):
        if (x - 1, y - 1) in self.wall_locs:
            self.wall_locs.remove((x - 1, y - 1))
            self.n_clutter_placed -= 1
        obj = self.grid.get(x, y)
        if obj is not None and obj.type == 'wall':
            self.grid.set(x, y, None)

    def _count_walls(self):
        wall_mask = np.array(
            [1 if isinstance(x, minigrid.Wall) else 0 for x in self.grid.grid],
            dtype=bool).reshape(self.height, self.width)[1:-1, 1:-1]
        self.wall_mask = wall_mask
        num_walls = wall_mask.sum()
        wall_pos = list(zip(*np.nonzero(wall_mask)))
        self.wall_locs = [(x + 1, y + 1) for y, x in wall_pos]
        for y, x in wall_pos:
            self.graph.remove_node((x, y))
        return num_walls

    def compute_shortest_path(self):
        if self.agent_start_pos is None or self.goal_pos is None:
            return
        self.distance_to_goal = (abs(self.goal_pos[0] - self.agent_start_pos[0]) +
                                  abs(self.goal_pos[1] - self.agent_start_pos[1]))
        self.passable = nx.has_path(
            self.graph,
            source=(self.agent_start_pos[0] - 1, self.agent_start_pos[1] - 1),
            target=(self.goal_pos[0] - 1, self.goal_pos[1] - 1))
        if self.passable:
            self.shortest_path_length = nx.shortest_path_length(
                self.graph,
                source=(self.agent_start_pos[0] - 1, self.agent_start_pos[1] - 1),
                target=(self.goal_pos[0] - 1, self.goal_pos[1] - 1))
        else:
            self.shortest_path_length = (self.width - 2) * (self.height - 2) + 1

    def generate_random_z(self):
        return np.random.uniform(size=(self.random_z_dim,)).astype(np.float32)

    def step_adversary(self, loc):
        if loc >= self.adversary_action_dim:
            raise ValueError('Position passed to step_adversary is outside the grid.')

        if self.resample_n_clutter and not self.n_clutter_sampled:
            n_clutter = int((loc / self.adversary_action_dim) * self.n_clutter)
            self.adversary_max_steps = n_clutter + 2
            self.n_clutter_sampled = True

        if self.adversary_step_count < self.adversary_max_steps:
            x = int(loc % (self.width - 2)) + 1
            y = int(loc / (self.width - 2)) + 1
            done = False

            if self.choose_goal_last:
                should_choose_goal = self.adversary_step_count == self.adversary_max_steps - 2
                should_choose_agent = self.adversary_step_count == self.adversary_max_steps - 1
            else:
                should_choose_goal = self.adversary_step_count == 0
                should_choose_agent = self.adversary_step_count == 1

            if should_choose_goal:
                if random.random() < self.goal_noise:
                    self.goal_pos = self.place_obj(minigrid.Goal(), max_tries=100)
                else:
                    self.remove_wall(x, y)
                    self.put_obj(minigrid.Goal(), x, y)
                    self.goal_pos = (x, y)
            elif should_choose_agent:
                self.remove_wall(x, y)
                if self.grid.get(x, y) is not None:
                    self.agent_start_pos = self.place_one_agent(0, rand_dir=False)
                    self.deliberate_agent_placement = 0
                else:
                    self.agent_start_pos = np.array([x, y])
                    self.place_agent_at_pos(0, self.agent_start_pos, rand_dir=False)
                    self.deliberate_agent_placement = 1
            elif self.adversary_step_count < self.adversary_max_steps:
                if self.grid.get(x, y) is None:
                    self.put_obj(minigrid.Wall(), x, y)
                    self.n_clutter_placed += 1
                    self.wall_locs.append((x - 1, y - 1))

        self.adversary_step_count += 1

        if self.adversary_step_count >= self.n_clutter + 2:
            done = True
            self.reset_metrics()
            self.compute_metrics()
        else:
            done = False

        image = self.grid.encode()
        obs = {
            'image': image,
            'time_step': [self.adversary_step_count],
            'random_z': self.generate_random_z(),
        }
        return obs, 0, done, {}

    def reset_random(self):
        if self.fixed_environment:
            self.seed(self.seed_value)

        self.graph = grid_graph(dim=[self.width - 2, self.height - 2])
        self.step_count = 0
        self.adversary_step_count = 0
        self.reset_agent_status()
        self.agent_start_pos = None
        self.goal_pos = None
        self.reset_metrics()
        self._gen_grid(self.width, self.height)

        self.goal_pos = self.place_obj(minigrid.Goal(), max_tries=100)
        self.agent_start_dir = self._rand_int(0, 4)
        self.agent_start_pos = self.place_one_agent(0, rand_dir=False)

        if self.resample_n_clutter:
            n_clutter = self._resample_n_clutter()
        else:
            n_clutter = int(self.n_clutter / 2)

        for _ in range(n_clutter):
            self.place_obj(minigrid.Wall(), max_tries=100)

        self.compute_metrics()
        return self.reset_agent()


class MiniAdversarialEnv(AdversarialEnv):
    def __init__(self):
        super().__init__(n_clutter=7, size=6, agent_view_size=5, max_steps=50)


class NoisyAdversarialEnv(AdversarialEnv):
    def __init__(self):
        super().__init__(goal_noise=0.3)


class MediumAdversarialEnv(AdversarialEnv):
    def __init__(self):
        super().__init__(n_clutter=30, size=10, agent_view_size=5, max_steps=200)


class GoalLastAdversarialEnv(AdversarialEnv):
    def __init__(self, fixed_environment=False, seed=None):
        super().__init__(choose_goal_last=True, fixed_environment=fixed_environment,
                         seed=seed or 0, max_steps=250)


class GoalLastAdversarialEnv30(AdversarialEnv):
    def __init__(self, fixed_environment=False, seed=None):
        super().__init__(choose_goal_last=True, n_clutter=30,
                         fixed_environment=fixed_environment, seed=seed or 0, max_steps=250)


class GoalLastAdversarialEnv60(AdversarialEnv):
    def __init__(self, fixed_environment=False, seed=None):
        super().__init__(choose_goal_last=True, n_clutter=60,
                         fixed_environment=fixed_environment, seed=seed or 0, max_steps=250)


class GoalLastOpaqueWallsAdversarialEnv(AdversarialEnv):
    def __init__(self, fixed_environment=False, seed=None):
        super().__init__(choose_goal_last=True, see_through_walls=False,
                         fixed_environment=fixed_environment, seed=seed or 0, max_steps=250)


class GoalLastFewerBlocksAdversarialEnv(AdversarialEnv):
    def __init__(self, fixed_environment=False, seed=None):
        super().__init__(choose_goal_last=True, n_clutter=25,
                         fixed_environment=fixed_environment, seed=seed or 0, max_steps=250)


class GoalLastVariableBlocksAdversarialEnv(AdversarialEnv):
    def __init__(self, fixed_environment=False, seed=None):
        super().__init__(choose_goal_last=True, n_clutter=60, resample_n_clutter=True,
                         fixed_environment=fixed_environment, seed=seed or 0, max_steps=250)


class GoalLastVariableBlocksAdversarialEnv_WNG(AdversarialEnv):
    def __init__(self, fixed_environment=False, seed=None):
        super().__init__(choose_goal_last=True, n_clutter=60, resample_n_clutter=True,
                         fixed_environment=fixed_environment, seed=seed or 0, max_steps=250,
                         editor_actions='walls_none_goal')


class MiniGoalLastAdversarialEnv(AdversarialEnv):
    def __init__(self, fixed_environment=False, seed=None):
        super().__init__(n_clutter=7, size=6, agent_view_size=5, max_steps=50,
                         choose_goal_last=True, fixed_environment=fixed_environment, seed=seed or 0)


if hasattr(__loader__, 'name'):
    module_path = __loader__.name
elif hasattr(__loader__, 'fullname'):
    module_path = __loader__.fullname

register.register(
    env_id='MultiGrid-Adversarial-v0',
    entry_point=module_path + ':AdversarialEnv',
    max_episode_steps=250,
)
register.register(
    env_id='MultiGrid-GoalLastAdversarial-v0',
    entry_point=module_path + ':GoalLastAdversarialEnv',
    max_episode_steps=250,
)
register.register(
    env_id='MultiGrid-GoalLastAdversarialEnv30-v0',
    entry_point=module_path + ':GoalLastAdversarialEnv30',
    max_episode_steps=250,
)
register.register(
    env_id='MultiGrid-GoalLastAdversarialEnv60-v0',
    entry_point=module_path + ':GoalLastAdversarialEnv60',
    max_episode_steps=250,
)
register.register(
    env_id='MultiGrid-GoalLastOpaqueWallsAdversarial-v0',
    entry_point=module_path + ':GoalLastOpaqueWallsAdversarialEnv',
    max_episode_steps=250,
)
register.register(
    env_id='MultiGrid-GoalLastFewerBlocksAdversarial-v0',
    entry_point=module_path + ':GoalLastFewerBlocksAdversarialEnv',
    max_episode_steps=250,
)
register.register(
    env_id='MultiGrid-GoalLastVariableBlocksAdversarialEnv-v0',
    entry_point=module_path + ':GoalLastVariableBlocksAdversarialEnv',
    max_episode_steps=250,
)
register.register(
    env_id='MultiGrid-GoalLastVariableBlocksAdversarialEnv-Edit-v0',
    entry_point=module_path + ':GoalLastVariableBlocksAdversarialEnv_WNG',
    max_episode_steps=250,
)
register.register(
    env_id='MultiGrid-MiniGoalLastAdversarial-v0',
    entry_point=module_path + ':MiniGoalLastAdversarialEnv',
    max_episode_steps=50,
)
