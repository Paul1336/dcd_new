import cv2
import random
import gymnasium as gym
from gymnasium.spaces import Discrete, Box
from iphyre.simulator import IPHYRE
import numpy as np
from iphyre.games import GAMES, PARAS


class IPHYRE_inadvance(gym.Env):
    """
    Plan in advance: generate one-step action times based on the initial scenes.
    """

    def __init__(self, config):
        self.total_reward = 0
        self.game_list = config.get("game_list")
        self.seed = config.get("seed")
        random.seed(self.seed)
        self.game = None
        self.env = None
        self.action_list = None
        self.action_candidates = None
        self.cur_obs = None
        self.fps = 10
        self.game_time = 15.0
        self.iter_len = 0
        self.action_space = Box(
            low=0.0, high=1.0, shape=(config.get("action_space", 6),), dtype=np.float32
        )
        self.observation_space = Box(
            low=0.0, high=1.0, shape=(12 * 9 + 6 * 2,), dtype=np.float32
        )
        self.reset_num = 0

    def reset(self, *, seed=None, options=None):
        self.iter_len = 0
        self.game = random.choice(self.game_list)
        self.env = IPHYRE(game=self.game, fps=self.fps)
        self.total_reward = 0
        self.cur_obs = self.env.reset()
        self.action_list = self.env.get_action_space()[1:]
        self.action_candidates = (
            np.array(self.action_list, dtype=np.float32).reshape(-1) / 600
        )
        self.process()
        self.reset_num += 1
        return self.cur_obs, {}

    def step(self, action):
        """
        :param action: the time sequence of executing each action (1 * 6)
        """
        total_reward = 0
        terminated = False
        truncated = True
        tmp = np.round(action * self.game_time, 1)
        for time in np.round(np.arange(0, self.game_time, 1 / self.fps), 1):
            if time > 0.0 and time in tmp:
                id = np.argwhere(tmp == time)[0][0]
                pos = self.action_list[id]
            else:
                pos = [0.0, 0.0]
            self.cur_obs, reward, terminated = self.env.step(pos)
            total_reward += reward
            if terminated:
                truncated = False
                break
        self.process()
        return self.cur_obs, total_reward, terminated, truncated, {}

    def process(self):
        self.cur_obs = np.array(self.cur_obs, dtype=np.float32)
        self.cur_obs[:, :5] /= 600
        self.cur_obs = self.cur_obs.clip(0.0, 1.0)
        self.cur_obs = self.cur_obs.reshape(-1)
        self.cur_obs = np.concatenate((self.cur_obs, self.action_candidates))


class IPHYRE_ontheflypenality(gym.Env):
    """
    Plan on-the-fly: generate actions step by step based on the intermediate state.
    """

    def __init__(self, config):
        self.total_reward = 0
        self.game_list = config.get("game_list")
        self.seed = config.get("seed")
        random.seed(self.seed)
        self.game = None
        self.env = None
        self.action_list = None
        self.action_candidates = None
        self.cur_obs = None
        self.fps = 10
        self.iter_len = 0
        self.action_space = Discrete(7)
        self.observation_space = Box(
            low=0.0, high=1.0, shape=(12 * 9 + 7 * 2,), dtype=np.float32
        )
        self.reset_num = 0

        # To penalize the agent for using the same action multiple times, we need to keep track of the used actions.
        self.used_action_list = []

    def reset(self, *, seed=None, options=None):
        self.iter_len = 0
        self.game = random.choice(self.game_list)
        self.env = IPHYRE(game=self.game, fps=self.fps)
        self.total_reward = 0
        self.cur_obs = self.env.reset()
        self.action_list = self.env.get_action_space()
        self.action_candidates = (
            np.array(self.action_list, dtype=np.float32).reshape(-1) / 600
        )
        self.process()
        self.reset_num += 1

        self.used_action_list = []
        return self.cur_obs, {}

    def step(self, action):
        repetitive_action_penality = -100
        should_penalize = False
        if action in self.used_action_list and action != 0:
            should_penalize = True
        if action != 0:
            self.used_action_list.append(action)

        self.iter_len += 1
        pos = self.action_list[action]
        self.cur_obs, reward, terminated = self.env.step(pos)
        self.process()
        truncated = self.iter_len >= 15 * self.fps
        reward += repetitive_action_penality if should_penalize else 0
        self.total_reward += reward

        return self.cur_obs, reward, terminated, truncated, {}

    def process(self):
        self.cur_obs = np.array(self.cur_obs, dtype=np.float32)
        self.cur_obs[:, :5] /= 600
        self.cur_obs = self.cur_obs.clip(0.0, 1.0)
        self.cur_obs = self.cur_obs.reshape(-1)
        self.cur_obs = np.concatenate((self.cur_obs, self.action_candidates))


import math
import pymunk


def distance_between_ball_and_segment(ball_body, segment_body):

    ball_pos = ball_body.position
    segment_shape = list(segment_body.shapes)[0]

    # x, y = segment_body.position
    # a_x, a_y = segment_shape.a
    # b_x, b_y = segment_shape.b

    # x1, y1 = x + a_x, y + a_y
    # x2, y2 = x + b_x, y + b_y
    angle = segment_body.angle

    a_x, a_y = segment_shape.a
    b_x, b_y = segment_shape.b
    # We assume that a and b are symmetric. Their half_length is the same.
    half_length = (a_x**2 + a_y**2) ** 0.5

    x, y = segment_body.position
    original_angle = math.atan2(b_y, b_x)
    current_angle = original_angle + angle
    x1, y1 = x - half_length * math.cos(current_angle), y - half_length * math.sin(
        current_angle
    )
    x2, y2 = x + half_length * math.cos(current_angle), y + half_length * math.sin(
        current_angle
    )

    # print(segment_body.position, a_x, a_y, b_x, b_y)
    # print('angle: ', angle, 'original_angle: ', original_angle, 'current_angle: ', current_angle)
    # print(x1, y1, x2, y2)

    # raise ValueError('stop here')
    segment_a = pymunk.Vec2d(x1, y1)
    segment_b = pymunk.Vec2d(x2, y2)

    # print(f'segment_a: {segment_a}, segment_b: {segment_b}, ball_pos: {ball_pos}')

    # Find the closest point on the segment to the ball.
    segment_vec = segment_b - segment_a
    ball_vec = ball_pos - segment_a

    # Note that the projected point should not outside the segment. Otherwise the distance is not correct.
    proj_len = max(
        0, min(1, np.dot(ball_vec, segment_vec) / np.linalg.norm(segment_vec) ** 2)
    )
    closest_point = segment_a + proj_len * segment_vec
    distance = np.linalg.norm(closest_point - ball_pos)

    # print(f'Distance: {distance}, closest_point: {closest_point}, proj_len: {proj_len}')
    return distance


def get_segment_descriptor(segment_body):
    segment_shape = list(segment_body.shapes)[0]

    if type(segment_shape) == pymunk.Circle:
        return f"{segment_body.position}, {segment_shape.radius}"
    angle = segment_body.angle

    a_x, a_y = segment_shape.a
    b_x, b_y = segment_shape.b
    # We assume that a and b are symmetric. Their half_length is the same.
    half_length = (a_x**2 + a_y**2) ** 0.5

    x, y = segment_body.position
    original_angle = math.atan2(b_y, b_x)
    current_angle = original_angle + angle
    x1, y1 = x - half_length * math.cos(current_angle), y - half_length * math.sin(
        current_angle
    )
    x2, y2 = x + half_length * math.cos(current_angle), y + half_length * math.sin(
        current_angle
    )

    return f"{x1, y1}, {x2, y2}, {segment_body.angle}, {half_length}, {a_x, a_y}, {b_x, b_y}"


def check_is_key_state(env, prev_ball_pos):
    """
    For every ball, we check whether it attaches to dynamic or eliminate bodies.
    """
    is_key_state = False
    key_state_reason = None
    skip_ball_count = 0
    min_distance = 100000

    for i, body in enumerate(env.space.bodies[-env.num_ball :]):
        ball_index = i + len(env.space.bodies) - env.num_ball

        # print(f'ball position: {body.position} prev_ball_pos: {prev_ball_pos}')
        distance = (
            np.linalg.norm(body.position - prev_ball_pos)
            if prev_ball_pos is not None
            else 0
        )
        prev_ball_pos = body.position
        # if distance < 3:
        #     print('skip because of the distance', distance)
        #     skip_ball_count += 1
        #     continue

        if "joint" in PARAS[env.game].keys():
            for j_idx, pair in enumerate(env.joint, start=1):
                if ball_index in pair:
                    is_key_state = True
                    key_state_reason = (
                        f"ball {ball_index} is in joint {j_idx}, pair: {pair}"
                    )
                    break
        if "spring" in PARAS[env.game].keys():
            for s_idx, pair in enumerate(env.spring, start=1):
                if ball_index in pair:
                    is_key_state = True
                    key_state_reason = (
                        f"ball {ball_index} is in spring {s_idx}, pair: {pair}"
                    )
                    break

        # Note: Avoid use arbiter to determine key states -- if the ball is rolling on a dynamic block, sometimes it is close but not a collision.
        # It is better to use the direct distance to determine key states.

        non_ball_bodies = env.space.bodies[: -env.num_ball]
        for i, non_ball_body in enumerate(non_ball_bodies):
            can_be_eliminated = env.eli[i] == 1
            is_dynamic_body = env.dynamic[i] == 1

            # print('check bodies')

            if can_be_eliminated or is_dynamic_body:
                circle_shape = list(body.shapes)[0]

                distance = distance_between_ball_and_segment(body, non_ball_body)
                # print(f'i {i}, {can_be_eliminated}, {is_dynamic_body}, Distance {distance}, Ball radius: {circle_shape.radius}')

                segment_radius = 10

                min_distance = min(min_distance, distance)

                # print(f'i: {i}, distance: {distance}, circle_shape.radius: {circle_shape.radius}, segment_radius: {segment_radius}')

                # If the ball is close to the segment, it is a key state.
                # This is to keep the momentum of the ball.
                eps = 50
                if distance < circle_shape.radius + segment_radius + eps:
                    is_key_state = True
                    key_state_reason = f"collision between ball and segment, distance {distance}, segment: {get_segment_descriptor(non_ball_body)}, ball: {get_segment_descriptor(body)}"
                    break
                # else:
                # print(f'distance {distance}, segment: {get_segment_descriptor(non_ball_body)}')

    if skip_ball_count == env.num_ball:
        is_key_state = False

    return is_key_state, prev_ball_pos, min_distance


class IPHYRE_ontheflykeystate(gym.Env):
    """
    Plan on-the-fly: generate actions step by step based on the intermediate state.
    """

    def __init__(self, config):
        self.total_reward = 0
        self.game_list = config.get("game_list")
        self.seed = config.get("seed")
        self.should_use_key_state = config.get("should_use_key_state", True)
        self.skip_per_frame = config.get("skip_per_frame", 60)
        random.seed(self.seed)
        self.game = None
        self.env = None
        self.action_list = None
        self.action_candidates = None
        self.cur_obs = None
        self.fps = 60
        self.iter_len = 0
        self.action_space = Discrete(7)
        self.use_images = config.get("use_images", False)

        # print('self.use_images: ', self.use_images)
        # print('self.should_use_key_state: ', self.should_use_key_state)
        # print('self.skip_per_frame: ', self.skip_per_frame)

        if self.use_images:
            self.observation_space = Box(
                low=0, high=255, shape=(600, 600, 3), dtype=np.uint8
            )
        else:
            self.observation_space = Box(
                low=0.0, high=1.0, shape=(12 * 9 + 7 * 2,), dtype=np.float32
            )
        self.reset_num = 0

        # To avoid redundant calculation of key state if the ball moves just a little bit.
        self.prev_ball_pos = None

    def reset(self, *, seed=None, options=None):
        self.iter_len = 0
        self.game = random.choice(self.game_list)
        self.env = IPHYRE(game=self.game, fps=self.fps)
        if self.use_images:
            self.env.init_screen()

        self.total_reward = 0
        self.cur_obs = self.env.reset(use_images=self.use_images)
        self.action_list = self.env.get_action_space()
        self.action_candidates = (
            np.array(self.action_list, dtype=np.float32).reshape(-1) / 600
        )
        self.process()
        self.reset_num += 1
        return self.cur_obs, {}

    def step(self, action):
        self.iter_len += 1
        pos = self.action_list[action]

        self.cur_obs, reward, terminated = self.env.step(
            pos, use_images=self.use_images
        )
        self.total_reward += reward

        non_key_state_count = 0
        while (
            non_key_state_count < self.skip_per_frame
            and not terminated
            and self.iter_len > 2
        ):
            if self.should_use_key_state:
                is_key_state, prev_ball_pos, min_distance = check_is_key_state(
                    self.env, self.prev_ball_pos
                )

                self.prev_ball_pos = prev_ball_pos
                if is_key_state:
                    break

            self.cur_obs, reward, terminated = self.env.step(
                self.action_list[0], use_images=self.use_images
            )

            self.total_reward += reward
            self.iter_len += 1
            non_key_state_count += 1

            if terminated or self.iter_len >= 15 * self.fps:
                break

        self.process()
        truncated = self.iter_len >= 15 * self.fps

        if self.cur_obs.max() > 255 and self.use_images:
            print(self.env.game)
            print("cur_obs.max(): ", self.cur_obs.max())
            print("cur_obs.min(): ", self.cur_obs.min())
            print("cur_obs.shape: ", self.cur_obs.shape)
            exit(0)

        return self.cur_obs, reward, terminated, truncated, {}

    def process(self):
        if self.use_images:
            return
        self.cur_obs = np.array(self.cur_obs, dtype=np.float32)
        self.cur_obs[:, :5] /= 600
        self.cur_obs = self.cur_obs.clip(0.0, 1.0)
        self.cur_obs = self.cur_obs.reshape(-1)
        self.cur_obs = np.concatenate((self.cur_obs, self.action_candidates))


class IPHYRE_onthefly(gym.Env):
    """
    Plan on-the-fly: generate actions step by step based on the intermediate state.
    """

    def __init__(self, config):
        self.total_reward = 0
        self.game_list = config.get("game_list")
        self.task_sampler = config.get("task_sampler")
        self.seed = config.get("seed")
        # If we set the seed, the task sampler will be deterministic.
        # random.seed(self.seed)
        self.game = None
        self.env = None
        self.action_list = None
        self.action_candidates = None
        self.cur_obs = None
        # In fps = 10, we have observed that the block may cross another block, perhaps due to the game engine issue.
        # We have found that fps = 60, but to keep the author's code consistent, we still use 10.
        # To use proper value, use IPHYRE_ontheflykeystate with skip_per_frame = 6.

        # UPDATE 20250219: BREAKING CHANGE. WE USE FPS = 60 instead of 10.
        self.fps = config.get("fps", 60)
        self.iter_len = 0
        self.action_space = Discrete(config.get("action_space", 7))
        self.reset_num = 0
        self.action_repeat = config.get("action_repeat", 1)
        self.reward_scale = config.get("reward_scale", 1.0)
        self.state_type = config.get("state_type", "symbolic")
        self.should_show_window = config.get("should_show_window", False)
        self.should_permute_block_order = config.get(
            "should_permute_block_order", False
        )
        self.should_add_task_encoding = config.get("should_add_task_encoding", False)
        # During evaluation, we evaluate on a single env, but we need to obtain task encoding from all envs.
        self.all_envs = config.get("all_envs", None)
        self.frame_skip = config.get("frame_skip", 1)

        if self.state_type == "symbolic":
            if self.should_add_task_encoding:
                self.observation_space = Box(
                    low=0.0,
                    high=1.0,
                    shape=(12 * 9 + 7 * 2 + len(self.all_envs),),
                    dtype=np.float32,
                )
            else:
                self.observation_space = Box(
                    low=0.0, high=1.0, shape=(12 * 9 + 7 * 2,), dtype=np.float32
                )
        elif self.state_type == "image":
            import os

            if not self.should_show_window:
                os.environ["SDL_VIDEODRIVER"] = "dummy"

            # Note that this setup aligns with Atari
            # https://gymnasium.farama.org/v0.29.0/environments/atari/adventure/
            # The channel is in RGB order.
            self.observation_space = Box(
                low=0, high=255, shape=(224, 224, 3), dtype=np.uint8
            )

    def reset(self, *, seed=None, options=None):
        self.iter_len = 0
        self.game = self.task_sampler.sample()
        # self.game = random.choice(self.game_list)
        self.env = IPHYRE(
            game=self.game,
            fps=self.fps,
            state_type=self.state_type,
            frame_skip=self.frame_skip,
        )
        if self.state_type == "image":
            self.env.init_screen()
        self.total_reward = 0
        self.cur_obs = self.env.reset()
        self.action_list = self.env.get_action_space()
        self.action_candidates = (
            np.array(self.action_list, dtype=np.float32).reshape(-1) / 600
        )
        self.process()
        self.reset_num += 1
        return self.cur_obs, {}

    def step(self, action):
        curstep_total_reward = 0
        terminated = False
        truncated = False
        for _ in range(self.action_repeat):
            self.iter_len += 1
            pos = self.action_list[action]
            self.cur_obs, reward, terminated = self.env.step(pos)
            self.process()
            truncated = self.iter_len >= 15 * self.fps / self.frame_skip
            scaled_reward = reward * self.reward_scale
            self.total_reward += scaled_reward

            curstep_total_reward += scaled_reward
            if terminated or truncated:
                break

        return self.cur_obs, curstep_total_reward, terminated, truncated, {}

    def process(self):
        if self.state_type == "symbolic":
            self.cur_obs = np.array(self.cur_obs, dtype=np.float32)
            self.cur_obs[:, :5] /= 600

            if self.should_permute_block_order:
                self.cur_obs = self.cur_obs[np.random.permutation(len(self.cur_obs))]

            self.cur_obs = self.cur_obs.clip(0.0, 1.0)
            self.cur_obs = self.cur_obs.reshape(-1)

            if self.should_add_task_encoding:
                task_encoding = np.zeros(len(self.all_envs), dtype=np.float32)
                task_encoding[self.all_envs.index(self.game)] = 1
                self.cur_obs = np.concatenate(
                    (self.cur_obs, self.action_candidates, task_encoding)
                )
            else:
                self.cur_obs = np.concatenate((self.cur_obs, self.action_candidates))
        elif self.state_type == "image":
            # downscale to 224
            self.cur_obs = cv2.resize(self.cur_obs, (224, 224))


class IPHYRE_combine(gym.Env):
    """
    Combined strategy: generate one-step action times based on the initial scenes but update after each execution.
    """

    def __init__(self, config):
        self.total_reward = 0
        self.game_list = config.get("game_list")
        self.seed = config.get("seed")
        random.seed(self.seed)
        self.game = None
        self.env = None
        self.action_list = None
        self.action_candidates = None
        self.cur_obs = None
        self.fps = 10
        self.game_time = 15.0
        self.iter_len = 0
        self.action_space = Box(low=0.0, high=1.0, shape=(6,), dtype=np.float32)
        self.observation_space = Box(
            low=0.0, high=1.0, shape=(12 * 9 + 6 * 2,), dtype=np.float32
        )
        self.reset_num = 0
        self.mask = None

    def reset(self, *, seed=None, options=None):
        self.iter_len = 0
        self.game = random.choice(self.game_list)
        self.env = IPHYRE(game=self.game, fps=self.fps)
        self.total_reward = 0
        self.cur_obs = self.env.reset()
        self.action_list = self.env.get_action_space()[1:]
        self.action_candidates = (
            np.array(self.action_list, dtype=np.float32).reshape(-1) / 600
        )
        self.mask = np.ones((6,))
        self.process()
        self.reset_num += 1
        return self.cur_obs, {}

    def step(self, action):
        total_reward = 0
        terminated = False
        tmp = np.round(action * self.game_time, 1)
        for time in np.round(
            np.arange(self.iter_len / self.fps, self.game_time, 1 / self.fps), 1
        ):
            self.iter_len += 1
            truncated = self.iter_len >= 15 * self.fps
            if time > 0.0 and time in tmp:
                id = np.argwhere(tmp == time)[0][0]
                if self.mask[id]:
                    pos = self.action_list[id]
                    self.mask[id] = 0
                else:
                    pos = [0.0, 0.0]
            else:
                pos = [0.0, 0.0]
            self.cur_obs, reward, terminated = self.env.step(pos)
            total_reward += reward
            if terminated:
                truncated = False
                break
            if pos != [0.0, 0.0]:
                break
        self.process()
        return self.cur_obs, total_reward, terminated, truncated, {}

    def process(self):
        self.cur_obs = np.array(self.cur_obs, dtype=np.float32)
        self.cur_obs[:, :5] /= 600
        self.cur_obs = self.cur_obs.clip(0.0, 1.0)
        self.cur_obs = self.cur_obs.reshape(-1)
        self.cur_obs = np.concatenate((self.cur_obs, self.action_candidates))
