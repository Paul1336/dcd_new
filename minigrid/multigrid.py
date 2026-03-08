# coding=utf-8
# Copyright 2021 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Multi-agent version of the Grid and MultiGridEnv classes."""
import math

import gym
import gym_minigrid.minigrid as minigrid
import gym_minigrid.rendering as rendering
import numpy as np

from . import window

# Map of color names to RGB values
AGENT_COLOURS = [
    np.array([60, 182, 234]),   # Blue
    np.array([229, 52, 52]),    # Red
    np.array([144, 32, 249]),   # Purple
    np.array([69, 196, 60]),    # Green
    np.array([252, 227, 35]),   # Yellow
]

class WorldObj(minigrid.WorldObj):
    """Override MiniGrid base class to deal with Agent objects."""

    def __init__(self, obj_type, color=None):
        assert obj_type in minigrid.OBJECT_TO_IDX, obj_type
        self.type = obj_type
        if color:
            assert color in minigrid.COLOR_TO_IDX, color
            self.color = color
        self.contains = None
        self.init_pos = None
        self.cur_pos = None

    @staticmethod
    def decode(type_idx, color_idx, state):
        obj_type = minigrid.IDX_TO_OBJECT[type_idx]
        if obj_type != 'agent':
            color = minigrid.IDX_TO_COLOR[color_idx]

        if obj_type == 'empty' or obj_type == 'unseen':
            return None

        if obj_type == 'wall':
            v = minigrid.Wall(color)
        elif obj_type == 'floor':
            v = minigrid.Floor(color)
        elif obj_type == 'ball':
            v = minigrid.Ball(color)
        elif obj_type == 'key':
            v = minigrid.Key(color)
        elif obj_type == 'box':
            v = minigrid.Box(color)
        elif obj_type == 'door':
            is_open = state == 0
            is_locked = state == 2
            v = Door(color, is_open, is_locked)
        elif obj_type == 'goal':
            v = minigrid.Goal()
        elif obj_type == 'lava':
            v = minigrid.Lava()
        elif obj_type == 'agent':
            v = Agent(color_idx, state)
        else:
            assert False, "unknown object type in decode '%s'" % obj_type

        return v


class Door(minigrid.Door):
    """Extends minigrid Door class to multiple agents possibly carrying keys."""

    def toggle(self, env, pos, carrying):
        if self.is_locked:
            if isinstance(carrying, minigrid.Key) and carrying.color == self.color:
                self.is_locked = False
                self.is_open = True
                return True
            return False
        self.is_open = not self.is_open
        return True


class Agent(WorldObj):
    """Class to represent other agents existing in the world."""

    def __init__(self, agent_id, state):
        super(Agent, self).__init__('agent')
        self.agent_id = agent_id
        self.dir = state

    def can_contain(self):
        return True

    def encode(self):
        return (minigrid.OBJECT_TO_IDX[self.type], self.agent_id, self.dir)

    def render(self, img):
        tri_fn = rendering.point_in_triangle(
            (0.12, 0.19),
            (0.87, 0.50),
            (0.12, 0.81),
        )
        tri_fn = rendering.rotate_fn(
            tri_fn, cx=0.5, cy=0.5, theta=0.5 * math.pi * self.dir)
        color = AGENT_COLOURS[self.agent_id]
        rendering.fill_coords(img, tri_fn, color)


class Grid(minigrid.Grid):
    """Extends Grid class, overrides some functions to cope with multi-agent case."""

    @classmethod
    def render_tile(cls, obj, highlight=None, tile_size=minigrid.TILE_PIXELS,
                    subdivs=3, cell_type=None):
        if isinstance(highlight, list):
            key = (tuple(highlight), tile_size)
        else:
            key = (highlight, tile_size)
        key = obj.encode() + key if obj else key

        if key in cls.tile_cache:
            return cls.tile_cache[key]

        img = np.zeros(shape=(tile_size * subdivs, tile_size * subdivs, 3), dtype=np.uint8)
        rendering.fill_coords(img, rendering.point_in_rect(0, 0.031, 0, 1), (100, 100, 100))
        rendering.fill_coords(img, rendering.point_in_rect(0, 1, 0, 0.031), (100, 100, 100))

        if obj is not None and obj.type != 'agent':
            obj.render(img)

        if highlight and not (cell_type is not None and cell_type == 'wall'):
            if isinstance(highlight, list):
                for a, agent_highlight in enumerate(highlight):
                    if agent_highlight:
                        rendering.highlight_img(img, color=AGENT_COLOURS[a])
            else:
                rendering.highlight_img(img)

        if obj is not None and obj.type == 'agent':
            obj.render(img)

        img = rendering.downsample(img, subdivs)
        cls.tile_cache[key] = img
        return img

    def render(self, tile_size, highlight_mask=None):
        if highlight_mask is None:
            highlight_mask = np.zeros(shape=(self.width, self.height), dtype=np.bool)

        width_px = self.width * tile_size
        height_px = self.height * tile_size
        img = np.zeros(shape=(height_px, width_px, 3), dtype=np.uint8)

        for y in range(0, self.height):
            for x in range(0, self.width):
                cell = self.get(x, y)
                cell_type = cell.type if cell else None

                if isinstance(highlight_mask, list):
                    n_agents = len(highlight_mask)
                    highlights = [highlight_mask[a][x, y] for a in range(n_agents)]
                else:
                    highlights = highlight_mask[x, y]

                tile_img = Grid.render_tile(cell, highlight=highlights,
                                             tile_size=tile_size, cell_type=cell_type)

                ymin = y * tile_size
                ymax = (y + 1) * tile_size
                xmin = x * tile_size
                xmax = (x + 1) * tile_size
                img[ymin:ymax, xmin:xmax, :] = tile_img

        return img

    def set_encoding(self, encoding, multigrid_env=None):
        assert tuple(encoding.shape[:2]) == (self.height, self.width)
        for i in range(self.height):
            for j in range(self.width):
                v = WorldObj.decode(*encoding[i, j, :])
                if isinstance(v, Agent):
                    v.agent_id = 0
                    if multigrid_env:
                        multigrid_env.agent_start_pos = np.array((i, j), dtype=np.int64)
                elif isinstance(v, minigrid.Goal):
                    if multigrid_env:
                        multigrid_env.goal_pos = np.array((i, j), dtype=np.int64)
                self.set(i, j, v)

    @staticmethod
    def decode(array):
        width, height, channels = array.shape
        assert channels == 3
        vis_mask = np.ones(shape=(width, height), dtype=np.bool)
        grid = Grid(width, height)
        for i in range(width):
            for j in range(height):
                type_idx, color_idx, state = array[i, j]
                v = WorldObj.decode(type_idx, color_idx, state)
                grid.set(i, j, v)
                vis_mask[i, j] = (type_idx != minigrid.OBJECT_TO_IDX['unseen'])
        return grid, vis_mask

    def rotate_left(self):
        grid = Grid(self.height, self.width)
        for i in range(self.width):
            for j in range(self.height):
                v = self.get(i, j)
                if v is not None and v.type == 'agent':
                    v = Agent(v.agent_id, v.dir)
                    v.dir -= 1
                    if v.dir < 0:
                        v.dir += 4
                grid.set(j, grid.height - 1 - i, v)
        return grid

    def slice(self, top_x, top_y, width, height, agent_pos=None):
        grid = Grid(width, height)
        for j in range(0, height):
            for i in range(0, width):
                x = top_x + i
                y = top_y + j
                if x >= 0 and x < self.width and y >= 0 and y < self.height:
                    v = self.get(x, y)
                else:
                    v = minigrid.Wall()
                grid.set(i, j, v)
        return grid


class MultiGridEnv(minigrid.MiniGridEnv):
    """2D grid world game environment with multi-agent support."""

    def __init__(self, grid_size=None, width=None, height=None, max_steps=100,
                 see_through_walls=False, seed=52, agent_view_size=7, n_agents=3,
                 competitive=False, fixed_environment=False, minigrid_mode=False):
        if grid_size:
            assert width is None and height is None
            width = grid_size
            height = grid_size

        self.n_agents = n_agents
        self.competitive = competitive

        if self.n_agents == 1:
            self.competitive = True

        self.actions = MultiGridEnv.Actions
        self.agent_view_size = agent_view_size
        self.reward_range = (0, 1)

        self.direction_obs_space = gym.spaces.Box(
            low=0, high=3, shape=(self.n_agents,), dtype='uint8')

        self.minigrid_mode = minigrid_mode
        if self.minigrid_mode:
            assert self.n_agents == 1, 'Backwards compatibility with minigrid only possible with 1 agent'
            self.action_space = gym.spaces.Discrete(len(self.actions))
            self.image_obs_space = gym.spaces.Box(
                low=0, high=255,
                shape=(self.agent_view_size, self.agent_view_size, 3),
                dtype='uint8')
        else:
            self.action_space = gym.spaces.Box(
                low=0, high=len(self.actions) - 1,
                shape=(self.n_agents,), dtype='int64')
            self.image_obs_space = gym.spaces.Box(
                low=0, high=255,
                shape=(self.n_agents, self.agent_view_size, self.agent_view_size, 3),
                dtype='uint8')

        self.observation_space = gym.spaces.Dict(
            {'image': self.image_obs_space, 'direction': self.direction_obs_space})

        self.window = None
        self.width = width
        self.height = height
        self.max_steps = max_steps
        self.see_through_walls = see_through_walls

        self.agent_pos = [None] * self.n_agents
        self.agent_dir = [None] * self.n_agents
        self.done = [False] * self.n_agents

        self.seed(seed=seed)
        self.fixed_environment = fixed_environment
        self.reset()

    def seed(self, seed):
        super().seed(seed=seed)
        self.seed_value = seed
        return [seed]

    def reset(self):
        if self.fixed_environment:
            self.seed(self.seed_value)

        self.agent_pos = [None] * self.n_agents
        self.agent_dir = [None] * self.n_agents
        self.done = [False] * self.n_agents

        self._gen_grid(self.width, self.height)

        for a in range(self.n_agents):
            assert self.agent_pos[a] is not None
            assert self.agent_dir[a] is not None
            start_cell = self.grid.get(*self.agent_pos[a])
            assert (start_cell.type == 'agent' or
                    start_cell is None or start_cell.can_overlap())

        self.carrying = [None] * self.n_agents
        self.step_count = 0
        obs = self.gen_obs()
        return obs

    def place_obj(self, obj, top=None, size=None, reject_fn=None, max_tries=math.inf):
        if top is None:
            top = (0, 0)
        else:
            top = (max(top[0], 0), max(top[1], 0))

        if size is None:
            size = (self.grid.width, self.grid.height)

        num_tries = 0
        while True:
            if num_tries > max_tries:
                raise gym.error.RetriesExceededError('Rejection sampling failed in place_obj')
            num_tries += 1

            pos = np.array((
                self._rand_int(top[0], min(top[0] + size[0], self.grid.width)),
                self._rand_int(top[1], min(top[1] + size[1], self.grid.height))
            ))

            if self.grid.get(*pos) is not None:
                continue

            pos_no_good = False
            for a in range(self.n_agents):
                if np.array_equal(pos, self.agent_pos[a]):
                    pos_no_good = True
            if pos_no_good:
                continue

            if reject_fn and reject_fn(self, pos):
                continue

            break

        self.grid.set(pos[0], pos[1], obj)
        if obj is not None:
            obj.init_pos = pos
            obj.cur_pos = pos
        return pos

    def place_agent(self, top=None, size=None, rand_dir=True, max_tries=math.inf):
        for a in range(self.n_agents):
            self.place_one_agent(a, top=top, size=size, rand_dir=rand_dir, max_tries=math.inf)

    def place_one_agent(self, agent_id, top=None, size=None, rand_dir=True,
                        max_tries=math.inf, agent_obj=None):
        self.agent_pos[agent_id] = None
        pos = self.place_obj(None, top, size, max_tries=max_tries)
        self.place_agent_at_pos(agent_id, pos, agent_obj=agent_obj, rand_dir=rand_dir)
        return pos

    def place_agent_at_pos(self, agent_id, pos, agent_obj=None, rand_dir=True):
        self.agent_pos[agent_id] = pos
        if rand_dir:
            self.agent_dir[agent_id] = 0

        if not agent_obj:
            agent_obj = Agent(agent_id, self.agent_dir[agent_id])
            agent_obj.init_pos = pos
        else:
            agent_obj.dir = self.agent_dir[agent_id]
        agent_obj.cur_pos = pos
        self.grid.set(pos[0], pos[1], agent_obj)

    @property
    def dir_vec(self):
        for a in range(self.n_agents):
            assert self.agent_dir[a] >= 0 and self.agent_dir[a] < 4
        return [minigrid.DIR_TO_VEC[self.agent_dir[a]] for a in range(self.n_agents)]

    @property
    def right_vec(self):
        return [np.array((-dy, dx)) for (dx, dy) in self.dir_vec]

    @property
    def front_pos(self):
        front_pos = [None] * self.n_agents
        for a in range(self.n_agents):
            assert self.agent_pos[a] is not None and self.dir_vec[a] is not None
            front_pos[a] = self.agent_pos[a] + self.dir_vec[a]
        return front_pos

    def get_view_coords(self, i, j, agent_id):
        ax, ay = self.agent_pos[agent_id]
        dx, dy = self.dir_vec[agent_id]
        rx, ry = self.right_vec[agent_id]
        sz = self.agent_view_size
        hs = self.agent_view_size // 2
        tx = ax + (dx * (sz - 1)) - (rx * hs)
        ty = ay + (dy * (sz - 1)) - (ry * hs)
        lx = i - tx
        ly = j - ty
        vx = (rx * lx + ry * ly)
        vy = -(dx * lx + dy * ly)
        return vx, vy

    def get_view_exts(self, agent_id):
        if self.agent_dir[agent_id] == 0:
            top_x = self.agent_pos[agent_id][0]
            top_y = self.agent_pos[agent_id][1] - self.agent_view_size // 2
        elif self.agent_dir[agent_id] == 1:
            top_x = self.agent_pos[agent_id][0] - self.agent_view_size // 2
            top_y = self.agent_pos[agent_id][1]
        elif self.agent_dir[agent_id] == 2:
            top_x = self.agent_pos[agent_id][0] - self.agent_view_size + 1
            top_y = self.agent_pos[agent_id][1] - self.agent_view_size // 2
        elif self.agent_dir[agent_id] == 3:
            top_x = self.agent_pos[agent_id][0] - self.agent_view_size // 2
            top_y = self.agent_pos[agent_id][1] - self.agent_view_size + 1
        else:
            assert False, 'invalid agent direction'
        bot_x = top_x + self.agent_view_size
        bot_y = top_y + self.agent_view_size
        return (top_x, top_y, bot_x, bot_y)

    def relative_coords(self, x, y, agent_id):
        vx, vy = self.get_view_coords(x, y, agent_id)
        if (vx < 0 or vy < 0 or vx >= self.agent_view_size or vy >= self.agent_view_size):
            return None
        return vx, vy

    def in_view(self, x, y, agent_id):
        return self.relative_coords(x, y, agent_id) is not None

    def agent_is_done(self, agent_id):
        pos = self.agent_pos[agent_id]
        agent_obj = self.grid.get(pos[0], pos[1])
        self.grid.set(pos[0], pos[1], None)
        self.done[agent_id] = True
        if self.carrying[agent_id]:
            self.place_obj(obj=self.carrying[agent_id])
            self.carrying[agent_id] = None
        self.place_one_agent(agent_id, agent_obj=agent_obj)

    def move_agent(self, agent_id, new_pos):
        old_pos = self.agent_pos[agent_id]
        agent_obj = self.grid.get(old_pos[0], old_pos[1])
        assert agent_obj.agent_id == agent_id
        assert (agent_obj.cur_pos == old_pos).all()
        self.grid.set(old_pos[0], old_pos[1], None)
        self.agent_pos[agent_id] = new_pos
        agent_obj.cur_pos = new_pos
        self.grid.set(new_pos[0], new_pos[1], agent_obj)

    def rotate_agent(self, agent_id):
        pos = self.agent_pos[agent_id]
        agent_obj = self.grid.get(pos[0], pos[1])
        assert agent_obj.agent_id == agent_id
        agent_obj.dir = self.agent_dir[agent_id]
        self.grid.set(pos[0], pos[1], agent_obj)

    def step_one_agent(self, action, agent_id):
        reward = 0
        fwd_pos = self.front_pos[agent_id]
        fwd_cell = self.grid.get(*fwd_pos)

        if action == self.actions.left:
            self.agent_dir[agent_id] -= 1
            if self.agent_dir[agent_id] < 0:
                self.agent_dir[agent_id] += 4
            self.rotate_agent(agent_id)
        elif action == self.actions.right:
            self.agent_dir[agent_id] = (self.agent_dir[agent_id] + 1) % 4
            self.rotate_agent(agent_id)
        elif action == self.actions.forward:
            agent_blocking = False
            for a in range(self.n_agents):
                if a != agent_id and np.array_equal(self.agent_pos[a], fwd_pos):
                    agent_blocking = True
            if not agent_blocking:
                if fwd_cell is not None and fwd_cell.type == 'goal':
                    self.agent_is_done(agent_id)
                    reward = self._reward()
                elif fwd_cell is not None and fwd_cell.type == 'lava':
                    self.agent_is_done(agent_id)
                elif fwd_cell is None or fwd_cell.can_overlap():
                    self.move_agent(agent_id, fwd_pos)
        elif action == self.actions.pickup:
            if fwd_cell and fwd_cell.can_pickup():
                if self.carrying[agent_id] is None:
                    self.carrying[agent_id] = fwd_cell
                    self.carrying[agent_id].cur_pos = np.array([-1, -1])
                    self.grid.set(fwd_pos[0], fwd_pos[1], None)
                    a_pos = self.agent_pos[agent_id]
                    agent_obj = self.grid.get(a_pos[0], a_pos[1])
                    agent_obj.contains = fwd_cell
        elif action == self.actions.drop:
            if not fwd_cell and self.carrying[agent_id]:
                self.grid.set(fwd_pos[0], fwd_pos[1], self.carrying[agent_id])
                self.carrying[agent_id].cur_pos = fwd_pos
                self.carrying[agent_id] = None
                a_pos = self.agent_pos[agent_id]
                agent_obj = self.grid.get(a_pos[0], a_pos[1])
                agent_obj.contains = None
        elif action == self.actions.toggle:
            if fwd_cell:
                if fwd_cell.type == 'door':
                    fwd_cell.toggle(self, fwd_pos, self.carrying[agent_id])
                else:
                    fwd_cell.toggle(self, fwd_pos)
        elif action == self.actions.done:
            pass
        else:
            assert False, 'unknown action'

        return reward

    def step(self, actions):
        if not isinstance(actions, list) and self.n_agents == 1:
            actions = [actions]

        self.step_count += 1
        rewards = [0] * self.n_agents

        agent_ordering = np.arange(self.n_agents)
        np.random.shuffle(agent_ordering)

        for a in agent_ordering:
            rewards[a] = self.step_one_agent(actions[a], a)

        obs = self.gen_obs()

        if self.minigrid_mode:
            rewards = rewards[0]

        collective_done = False
        if self.competitive:
            collective_done = np.sum(self.done) >= 1
        if self.step_count >= self.max_steps:
            collective_done = True

        return obs, rewards, collective_done, {}

    def gen_obs_grid(self, agent_id):
        top_x, top_y, _, _ = self.get_view_exts(agent_id)
        grid = self.grid.slice(top_x, top_y, self.agent_view_size, self.agent_view_size)

        for _ in range(self.agent_dir[agent_id] + 1):
            grid = grid.rotate_left()

        if not self.see_through_walls:
            vis_mask = grid.process_vis(
                agent_pos=(self.agent_view_size // 2, self.agent_view_size - 1))
        else:
            vis_mask = np.ones(shape=(grid.width, grid.height), dtype=np.bool)

        agent_pos = grid.width // 2, grid.height - 1
        if self.carrying[agent_id]:
            grid.set(agent_pos[0], agent_pos[1], self.carrying[agent_id])
        else:
            grid.set(agent_pos[0], agent_pos[1], None)

        return grid, vis_mask

    def gen_obs(self):
        images = []
        dirs = []
        for a in range(self.n_agents):
            image, direction = self.gen_agent_obs(a)
            images.append(image)
            dirs.append(direction)

        if self.minigrid_mode:
            images = images[0]

        obs = {'image': images, 'direction': dirs}
        return obs

    def gen_agent_obs(self, agent_id):
        grid, vis_mask = self.gen_obs_grid(agent_id)
        image = grid.encode(vis_mask)
        return image, self.agent_dir[agent_id]

    def compute_agent_visibility_mask(self, agent_id):
        highlight_mask = np.zeros(shape=(self.width, self.height), dtype=np.bool)
        _, vis_mask = self.gen_obs_grid(agent_id)
        f_vec = self.dir_vec[agent_id]
        r_vec = self.right_vec[agent_id]
        top_left = (self.agent_pos[agent_id] + f_vec * (self.agent_view_size - 1)
                    - r_vec * (self.agent_view_size // 2))
        for vis_j in range(0, self.agent_view_size):
            for vis_i in range(0, self.agent_view_size):
                if not vis_mask[vis_i, vis_j]:
                    continue
                abs_i, abs_j = top_left - (f_vec * vis_j) + (r_vec * vis_i)
                if abs_i < 0 or abs_i >= self.width:
                    continue
                if abs_j < 0 or abs_j >= self.height:
                    continue
                highlight_mask[abs_i, abs_j] = True
        return highlight_mask

    def render(self, mode='human', close=False, highlight=True, tile_size=minigrid.TILE_PIXELS):
        if close:
            if self.window:
                self.window.close()
            return None

        if mode == 'human' and not self.window:
            self.window = window.Window('gym_minigrid')
            self.window.show(block=False)

        if highlight:
            highlight_mask = []
            for a in range(self.n_agents):
                if self.agent_pos[a] is not None:
                    highlight_mask.append(self.compute_agent_visibility_mask(a))
        else:
            highlight_mask = None

        img = self.grid.render(tile_size, highlight_mask=highlight_mask)

        if mode == 'human':
            self.window.show_img(img)
            if hasattr(self, 'mission'):
                self.window.set_caption(self.mission)
            self.window.show()

        return img
