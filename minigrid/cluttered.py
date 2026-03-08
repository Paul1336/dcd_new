# coding=utf-8
# Copyright 2021 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Multi-agent goal-seeking task with many static obstacles."""

import gym_minigrid.minigrid as minigrid
from . import multigrid
from . import register


class ClutteredMultiGrid(multigrid.MultiGridEnv):
    """Goal seeking environment with obstacles."""

    def __init__(self, size=15, n_agents=3, n_clutter=25, randomize_goal=True,
                 agent_view_size=5, max_steps=250, walls_are_lava=False, **kwargs):
        self.n_clutter = n_clutter
        self.randomize_goal = randomize_goal
        self.walls_are_lava = walls_are_lava
        super().__init__(grid_size=size, max_steps=max_steps, n_agents=n_agents,
                         agent_view_size=agent_view_size, **kwargs)

    def _gen_grid(self, width, height):
        self.grid = multigrid.Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)
        if self.randomize_goal:
            self.place_obj(minigrid.Goal(), max_tries=100)
        else:
            self.put_obj(minigrid.Goal(), width - 2, height - 2)
        for _ in range(self.n_clutter):
            if self.walls_are_lava:
                self.place_obj(minigrid.Lava(), max_tries=100)
            else:
                self.place_obj(minigrid.Wall(), max_tries=100)
        self.place_agent()
        self.mission = 'get to the green square'

    def step(self, action):
        obs, reward, done, info = multigrid.MultiGridEnv.step(self, action)
        return obs, reward, done, info


class Cluttered40Minigrid(ClutteredMultiGrid):
    def __init__(self):
        super().__init__(n_agents=1, n_clutter=40, minigrid_mode=True)


class Cluttered10Minigrid(ClutteredMultiGrid):
    def __init__(self):
        super().__init__(n_agents=1, n_clutter=10, minigrid_mode=True)


class Cluttered50Minigrid(ClutteredMultiGrid):
    def __init__(self):
        super().__init__(n_agents=1, n_clutter=50, minigrid_mode=True)


class Cluttered5Minigrid(ClutteredMultiGrid):
    def __init__(self):
        super().__init__(n_agents=1, n_clutter=5, minigrid_mode=True)


if hasattr(__loader__, 'name'):
    module_path = __loader__.name
elif hasattr(__loader__, 'fullname'):
    module_path = __loader__.fullname

register.register(
    env_id='MultiGrid-Cluttered40-Minigrid-v0',
    entry_point=module_path + ':Cluttered40Minigrid',
)
register.register(
    env_id='MultiGrid-Cluttered10-Minigrid-v0',
    entry_point=module_path + ':Cluttered10Minigrid',
)
register.register(
    env_id='MultiGrid-Cluttered50-Minigrid-v0',
    entry_point=module_path + ':Cluttered50Minigrid',
)
register.register(
    env_id='MultiGrid-Cluttered5-Minigrid-v0',
    entry_point=module_path + ':Cluttered5Minigrid',
)
