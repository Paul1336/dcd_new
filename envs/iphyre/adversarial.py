# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import random
import json
import os

import gym
import time
import numpy as np
import torch
import cv2
from iphyre.simulator import IPHYRE
from iphyre.simulator import PARAS
import copy

from gym.spaces import Discrete, Box

from envs.registration import register as gym_register

def rand_int_seed():
    return str(int.from_bytes(os.urandom(4), byteorder="little"))

standard_task_config = {
    "block": [
        [[100.0, 150.0], [330.0, 200.0]],
        [[160.0, 100.0], [160.0, 130.0]],
        [[100.0, 400.0], [250.0, 400.0]],
        [[350.0, 400.0], [500.0, 400.0]],
        [[500.0, 300.0], [500.0, 380.0]],
        [[250.0, 360.0], [250.0, 380.0]],
        [[350.0, 360.0], [350.0, 380.0]],
        [[100.0, 300.0], [100.0, 380.0]],
    ],
    "ball": [[120.0, 120.0, 20.0]],
    "eli": [1, 1, 0, 0, 0, 0, 0, 0, 0],
    "dynamic": [0, 0, 0, 0, 0, 0, 0, 0, 1],
}

def edit_task(num_edits=1, reference_task_config=None):
    total_tries = 0

    while True:
        total_tries += 1
        task_config = copy.deepcopy(reference_task_config)

        if total_tries > 50:
            return reference_task_config


        for _ in range(num_edits):
            num_block_and_ball = len(task_config["block"]) + len(task_config["ball"])
            index = np.random.randint(0, num_block_and_ball)

            offset_range = 30
            if index < len(task_config["block"]):
                offset_x1 = np.random.randint(-offset_range, offset_range)
                offset_y1 = np.random.randint(-offset_range, offset_range)
                offset_x2 = np.random.randint(-offset_range, offset_range)
                offset_y2 = np.random.randint(-offset_range, offset_range)
                task_config["block"][index][0][0] += offset_x1
                task_config["block"][index][0][1] += offset_y1
                task_config["block"][index][1][0] += offset_x2
                task_config["block"][index][1][1] += offset_y2

                flip_eliminate = np.random.rand() < 0.5
                if flip_eliminate:
                    task_config["eli"][index] = 1 - task_config["eli"][index]
            else:
                offset_x = np.random.randint(-offset_range, offset_range)
                offset_y = np.random.randint(-offset_range, offset_range)
                task_config["ball"][0][0] += offset_x
                task_config["ball"][0][1] += offset_y

        total_eliminated = sum(task_config["eli"])
        if total_eliminated > 6:
            print("Too many blocks eliminated, skipping", task_config)
            continue
        
        # Note that the ball has 20 radius.
        is_outside_boundary = (
            task_config["ball"][0][0] < 20
            or task_config["ball"][0][0] > 580
            or task_config["ball"][0][1] < 20
            or task_config["ball"][0][1] > 580
        )
        
        is_all_good_blocks = True
        for block in task_config["block"]:
            length = np.linalg.norm(np.array(block[0]) - np.array(block[1]))
            if length < 1:
                is_all_good_blocks = False
                break

        if not is_all_good_blocks:
            print("Some blocks are too short, skipping", task_config)
            continue
        
        elif is_outside_boundary:
            print("Ball is outside boundary, skipping", task_config)
            continue
        else:
            return task_config

def generate_task(eval_type="random"):
    global standard_task_config

    offset_range = 100
    mutation_threshold = 0.3

    total_tries = 0

    while True:
        sample_task_config = copy.deepcopy(standard_task_config)
        total_tries += 1
        if total_tries > 50:
            return sample_task_config

        if eval_type == "random":
            # randomly pick a block or ball to modify its location
            for index in range(
                len(sample_task_config["block"]) + len(sample_task_config["ball"])
            ):
                should_mutate = np.random.rand() < mutation_threshold

                if not should_mutate:
                    continue

                if index < len(sample_task_config["block"]):
                    offset_x1 = np.random.randint(-offset_range, offset_range)
                    offset_y1 = np.random.randint(-offset_range, offset_range)
                    offset_x2 = np.random.randint(-offset_range, offset_range)
                    offset_y2 = np.random.randint(-offset_range, offset_range)
                    sample_task_config["block"][index][0][0] += offset_x1
                    sample_task_config["block"][index][0][1] += offset_y1
                    sample_task_config["block"][index][1][0] += offset_x2
                    sample_task_config["block"][index][1][1] += offset_y2
                else:
                    offset_x = np.random.randint(-offset_range, offset_range)
                    offset_y = np.random.randint(-offset_range, offset_range)
                    sample_task_config["ball"][0][0] += offset_x
                    sample_task_config["ball"][0][1] += offset_y

        # Note that the ball has 20 radius.
        is_outside_boundary = (
            sample_task_config["ball"][0][0] < 20
            or sample_task_config["ball"][0][0] > 580
            or sample_task_config["ball"][0][1] < 20
            or sample_task_config["ball"][0][1] > 580
        )
        
        is_all_good_blocks = True
        for block in sample_task_config["block"]:
            length = np.linalg.norm(np.array(block[0]) - np.array(block[1]))
            if length < 1:
                is_all_good_blocks = False
                break

        if not is_all_good_blocks:
            print("Some blocks are too short, skipping", sample_task_config)
            continue
        
        if is_outside_boundary:
            print("Ball is outside boundary, skipping", sample_task_config)
            continue

        return sample_task_config



class IphyreAdversarialEnv(gym.Env):
    """
    Plan on-the-fly: generate actions step by step based on the intermediate state.
    """

    def __init__(self, config={}, seed=None, random_z_dim=10, reset_random_after_episode=False):
        self.total_reward = 0

        self.passable = True

        self.reset_random_after_episode = reset_random_after_episode

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
        
        self.adversary_step_count = 0
        self.random_z_dim = random_z_dim
        self.adversary_max_steps = 10
        self.adversary_action_dim = 1
        self.adversary_action_space = self.action_space
        self.adversary_ts_obs_space = gym.spaces.Box(
            low=0, high=self.adversary_max_steps, shape=(1,), dtype='uint8')
        self.adversary_randomz_obs_space = gym.spaces.Box(
            low=0, high=1.0, shape=(self.random_z_dim,), dtype=np.float32)
        self.adversary_image_obs_space = gym.spaces.Box(
            low=0,
            high=1.0,
            shape=(12 * 9 + 7 * 2,),
            dtype=np.float32)
        self.adversary_observation_space = gym.spaces.Dict({
            'image': self.adversary_image_obs_space,
            'time_step': self.adversary_ts_obs_space,
            'random_z': self.adversary_randomz_obs_space
        })
        
        self.reset_num = 0
        self.action_repeat = config.get("action_repeat", 4)
        self.reward_scale = config.get("reward_scale", 0.001)
        self.state_type = config.get("state_type", "symbolic")
        self.should_show_window = config.get("should_show_window", False)
        # During evaluation, we evaluate on a single env, but we need to obtain task encoding from all envs.
        self.all_envs = config.get("all_envs", None)
        self.frame_skip = config.get("frame_skip", 6)

        self.level_seed = 0
        global standard_task_config
        PARAS[self.level_seed] = standard_task_config

        self.task_config = standard_task_config

        if self.state_type == "symbolic":
            
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


    @property
    def processed_action_dim(self):
        return 1

    @property
    def encoding(self):
        enc = json.dumps(self.task_config) + '@@' + str(self.level_seed)
        return enc
    
    @property
    def level(self):
        return self.encoding
    
    def reset_to_level(self, level):
        # print(f"Resetting to level {level}")

        task_config = json.loads(level[:level.find('@@')])
        self.level_seed = level[level.find('@@')+2:]
        self.task_config = task_config

        PARAS[self.level_seed] = task_config

        return self.reset_agent()

    def generate_random_z(self):
        return np.random.uniform(size=(self.random_z_dim,)).astype(np.float32)

    def reset(self):
        self.adversary_step_count = 0

        if self.reset_random_after_episode:
            self.reset_to_random_level()

        obs = self.reset_agent()

        return obs

    def reset_to_random_level(self):
        self.level_seed = rand_int_seed()

        print('Reset Random: ', self.level_seed)

        # Generate a new task config    
        task_config = generate_task()

        # Write the game
        self.task_config = task_config
        PARAS[self.level_seed] = task_config

    def reset_random(self, *, seed=None, options=None):
        self.iter_len = 0
        self.reset_to_random_level()

        return self.reset_agent()

    def reset_agent(self):
        self.iter_len = 0
        self.env = IPHYRE(
            game=self.level_seed,
            fps=self.fps,
            state_type=self.state_type,
            frame_skip=self.frame_skip,
        )
        if self.state_type == "image":
            self.env.init_screen()
        self.total_reward = 0
        self.cur_obs = self.env.reset()
        self.cur_full_obs = PARAS[self.level_seed]
        self.action_list = self.env.get_action_space()
        self.action_candidates = (
            np.array(self.action_list, dtype=np.float32).reshape(-1) / 600
        )
        self.process()
        self.reset_num += 1
        return self.cur_obs
    
    def mutate_level(self, num_edits=1):
        if num_edits > 0:
            task_config = PARAS[self.level_seed]
            edited_task_config = edit_task(num_edits=num_edits, reference_task_config=task_config)

            new_level_seed = rand_int_seed()
            print(f"Mutated game {self.level_seed} to {new_level_seed} with {num_edits} edits")

            self.level_seed = new_level_seed
            PARAS[self.level_seed] = edited_task_config
            self.task_config = edited_task_config
            
        return self.reset_agent()


    def render(self, mode="human"):
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        self.env.init_screen()
        image = self.env.get_image_state()
        # Write level seed to bottom left corner using cv2
        image = np.ascontiguousarray(image)
        # Resize from 600x600 to 100x100
        image = cv2.resize(image, (300, 300))
        cv2.putText(image, str(self.level_seed), (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
        print('render image with shape', image.shape, self.level_seed)
        return image

    def step(self, action):
        # In eval, they send a list: [action] 
        if isinstance(action, list):
            # print('action is a list')
            action = action[0]
        if isinstance(action, torch.Tensor):
            # print('action is a tensor')
            action = action.item()  # Converts tensor([1]) to 1
        if isinstance(action, np.ndarray):
            # print('action is a numpy array')
            action = action[0]

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

        return self.cur_obs, curstep_total_reward, terminated or truncated, {}

    def process(self):
        if self.state_type == "symbolic":
            self.cur_obs = np.array(self.cur_obs, dtype=np.float32)
            self.cur_obs[:, :5] /= 600

            self.cur_obs = self.cur_obs.clip(0.0, 1.0)
            self.cur_obs = self.cur_obs.reshape(-1)

            self.cur_obs = np.concatenate((self.cur_obs, self.action_candidates))
        elif self.state_type == "image":
            # downscale to 224
            self.cur_obs = cv2.resize(self.cur_obs, (224, 224))


class IphyreGameEnv(IphyreAdversarialEnv):
    def __init__(self, env_name=None, config={}, seed=None, random_z_dim=10, env_task_config=None, reset_random_after_episode=False):
        super().__init__(config, seed, random_z_dim, reset_random_after_episode)
        self.level_seed = env_name
        if env_task_config is not None:
            PARAS[self.level_seed] = env_task_config
        # print(f"Loaded test task: {self.game}")
        # print("Task config: ", env_task_config)


def load_vlm_gen_tasks_solvable(
    task_dir_list, should_check_solvable
):
    env_ids = []
    generated_task_used_count = 0
    task_info_dict = {}
    for task_dir in task_dir_list:
        all_tasks = os.listdir(task_dir)
        for task in all_tasks:
            config_path = os.path.join(task_dir, task, "config.json")
            try:
                config = json.load(open(config_path, "r"))
                if not should_check_solvable or config["success_rate"] > 0.0:
                    env_ids.append(task)
                    PARAS[task] = config["config"]
                    generated_task_used_count += 1

                    if not "success_rate" in config:
                        config["success_rate"] = 0.0

                    task_info_dict[task] = {
                        "random_success_rate": config["success_rate"]
                    }
                    if "llm_eval_score" in config:
                        task_info_dict[task]["llm_eval_score"] = config["llm_eval_score"]
            except Exception as e:
                print(f"Error loading config for task {task}: {e}")

    print(f"Generated task used count: {generated_task_used_count}")
    return env_ids, task_info_dict


from envs.iphyre.datapath import vlm_10k_task_dir_list, vlm_4k_task_dir_list, vlm_10k_claude_task_dir_list, vlm_10k_gemini_task_dir_list

class IphyreAdversarialVLM4kEnv(IphyreAdversarialEnv):
    def __init__(self, config={}, seed=None, random_z_dim=10, reset_random_after_episode=False, env_names=[]):
        super().__init__(config, seed, random_z_dim, reset_random_after_episode)
        
        task_dir_list = vlm_4k_task_dir_list

        if len(env_names) != 4000:
            raise ValueError(f"Expected 4000 env names, but got {len(env_names)}")

        _, task_info_dict = load_vlm_gen_tasks_solvable(
            task_dir_list, should_check_solvable=False
        )


        self.subsampled_env_ids = env_names
        self.subsampled_task_info_dict = {
            env_id: task_info_dict[env_id] for env_id in self.subsampled_env_ids
        }

        print('AdversarialVLM4kEnv: First 10 env_names: ', self.subsampled_env_ids[:10])


    def reset_to_random_level(self):
        self.level_seed = random.choice(self.subsampled_env_ids)
        self.task_config = PARAS[self.level_seed]

        # print('Reset Random on VLM dataset: ', self.level_seed)


class IphyreAdversarialVLM10kEnv(IphyreAdversarialEnv):
    def __init__(self, config={}, seed=None, random_z_dim=10, reset_random_after_episode=False, env_names=[]):
        super().__init__(config, seed, random_z_dim, reset_random_after_episode)
        
        task_dir_list = vlm_10k_task_dir_list

        if len(env_names) != 10000:
            raise ValueError(f"Expected 10000 env names, but got {len(env_names)}")

        _, task_info_dict = load_vlm_gen_tasks_solvable(
            task_dir_list, should_check_solvable=False
        )

        self.subsampled_env_ids = env_names
        self.subsampled_task_info_dict = {
            env_id: task_info_dict[env_id] for env_id in self.subsampled_env_ids
        }
        print('AdversarialVLM10kEnv: First 10 env_names: ', self.subsampled_env_ids[:10])
    

    def reset_to_random_level(self):
        self.level_seed = random.choice(self.subsampled_env_ids)
        self.task_config = PARAS[self.level_seed]

        # print('Reset Random on VLM dataset: ', self.level_seed)




class IphyreAdversarialClaudeVLM10kEnv(IphyreAdversarialEnv):
    def __init__(self, config={}, seed=None, random_z_dim=10, reset_random_after_episode=False, env_names=[]):
        super().__init__(config, seed, random_z_dim, reset_random_after_episode)
        
        task_dir_list = vlm_10k_claude_task_dir_list

        if len(env_names) != 10000:
            raise ValueError(f"Expected 10000 env names, but got {len(env_names)}")

        _, task_info_dict = load_vlm_gen_tasks_solvable(
            task_dir_list, should_check_solvable=False
        )

        self.subsampled_env_ids = env_names
        self.subsampled_task_info_dict = {
            env_id: task_info_dict[env_id] for env_id in self.subsampled_env_ids
        }
        print('AdversarialClaudeVLM10kEnv: First 10 env_names: ', self.subsampled_env_ids[:10])
    

    def reset_to_random_level(self):
        self.level_seed = random.choice(self.subsampled_env_ids)
        self.task_config = PARAS[self.level_seed]

        # print('Reset Random on VLM dataset: ', self.level_seed)


class IphyreAdversarialGeminiVLM10kEnv(IphyreAdversarialEnv):
    def __init__(self, config={}, seed=None, random_z_dim=10, reset_random_after_episode=False, env_names=[]):
        super().__init__(config, seed, random_z_dim, reset_random_after_episode)
        
        task_dir_list = vlm_10k_gemini_task_dir_list

        if len(env_names) != 10000:
            raise ValueError(f"Expected 10000 env names, but got {len(env_names)}")

        _, task_info_dict = load_vlm_gen_tasks_solvable(
            task_dir_list, should_check_solvable=False
        )

        self.subsampled_env_ids = env_names
        self.subsampled_task_info_dict = {
            env_id: task_info_dict[env_id] for env_id in self.subsampled_env_ids
        }
        print('AdversarialGeminiVLM10kEnv: First 10 env_names: ', self.subsampled_env_ids[:10])
    

    def reset_to_random_level(self):
        self.level_seed = random.choice(self.subsampled_env_ids)
        self.task_config = PARAS[self.level_seed]

        # print('Reset Random on VLM dataset: ', self.level_seed)


class IphyreAdversarialProceduralRotateEnv(IphyreAdversarialEnv):
    def __init__(self, config={}, seed=None, random_z_dim=10, reset_random_after_episode=False):
        super().__init__(config, seed, random_z_dim, reset_random_after_episode)
        
        task_dir_list = [
            '../iphyre/test_toy20250110/20250602/output_eval_rotate'
        ]

        env_names, task_info_dict = load_vlm_gen_tasks_solvable(
            task_dir_list, should_check_solvable=True
        )

        self.subsampled_env_ids = env_names
        self.subsampled_task_info_dict = {
            env_id: task_info_dict[env_id] for env_id in self.subsampled_env_ids
        }
        print('AdversarialProceduralRotateEnv: First 10 env_names: ', self.subsampled_env_ids[:10])
    

    def reset_to_random_level(self):
        self.level_seed = random.choice(self.subsampled_env_ids)
        self.task_config = PARAS[self.level_seed]

        # print('Reset Random on VLM dataset: ', self.level_seed)

class IphyreAdversarialProceduralShiftEnv(IphyreAdversarialEnv):
    def __init__(self, config={}, seed=None, random_z_dim=10, reset_random_after_episode=False):
        super().__init__(config, seed, random_z_dim, reset_random_after_episode)
        
        task_dir_list = [
            '../iphyre/test_toy20250110/20250602/output_eval_shift'
        ]

        env_names, task_info_dict = load_vlm_gen_tasks_solvable(
            task_dir_list, should_check_solvable=True
        )


        self.subsampled_env_ids = env_names
        self.subsampled_task_info_dict = {
            env_id: task_info_dict[env_id] for env_id in self.subsampled_env_ids
        }
        print('AdversarialProceduralShiftEnv: First 10 env_names: ', self.subsampled_env_ids[:10])
    

    def reset_to_random_level(self):
        self.level_seed = random.choice(self.subsampled_env_ids)
        self.task_config = PARAS[self.level_seed]

        # print('Reset Random on VLM dataset: ', self.level_seed)


class IphyreAdversarialHandDesignSingleEnv(IphyreAdversarialEnv):
    def __init__(self, config={}, seed=None, random_z_dim=10, reset_random_after_episode=False, single_env_name=None):
        super().__init__(config, seed, random_z_dim, reset_random_after_episode)
        
        task_dir_list = [
            '../iphyre/test_toy20250110/20250525/output_hand_test'
        ]

        env_names, task_info_dict = load_vlm_gen_tasks_solvable(
            task_dir_list, should_check_solvable=False
        )

        self.subsampled_env_ids = [single_env_name]

        print('AdversarialHandDesignBlockAngleEnv', self.subsampled_env_ids)
    

    def reset_to_random_level(self):
        self.level_seed = random.choice(self.subsampled_env_ids)
        self.task_config = PARAS[self.level_seed]


# hole_block_angle
# hole_block_angle_hard
# hole_block_no_gap
# hole_block_order
# hole_confuse
# hole_double
# hole_down
# hole_high_ball
# hole_high_gap
# hole_outside
# hole_reverse
# hole_right
# hole_skip_gap
# hole_skip_gap_hard
# hole_swap

class IphyreAdversarialHandDesignBlockAngleEnv(IphyreAdversarialHandDesignSingleEnv):
    def __init__(self, config={}, seed=None, random_z_dim=10, reset_random_after_episode=False):
        super().__init__(config, seed, random_z_dim, reset_random_after_episode, single_env_name='hole_block_angle')

class IphyreAdversarialHandDesignBlockAngleHardEnv(IphyreAdversarialHandDesignSingleEnv):
    def __init__(self, config={}, seed=None, random_z_dim=10, reset_random_after_episode=False):
        super().__init__(config, seed, random_z_dim, reset_random_after_episode, single_env_name='hole_block_angle_hard')

class IphyreAdversarialHandDesignBlockNoGapEnv(IphyreAdversarialHandDesignSingleEnv):
    def __init__(self, config={}, seed=None, random_z_dim=10, reset_random_after_episode=False):
        super().__init__(config, seed, random_z_dim, reset_random_after_episode, single_env_name='hole_block_no_gap')

class IphyreAdversarialHandDesignBlockOrderEnv(IphyreAdversarialHandDesignSingleEnv):
    def __init__(self, config={}, seed=None, random_z_dim=10, reset_random_after_episode=False):
        super().__init__(config, seed, random_z_dim, reset_random_after_episode, single_env_name='hole_block_order')

class IphyreAdversarialHandDesignConfuseEnv(IphyreAdversarialHandDesignSingleEnv):
    def __init__(self, config={}, seed=None, random_z_dim=10, reset_random_after_episode=False):
        super().__init__(config, seed, random_z_dim, reset_random_after_episode, single_env_name='hole_confuse')

class IphyreAdversarialHandDesignDoubleEnv(IphyreAdversarialHandDesignSingleEnv):
    def __init__(self, config={}, seed=None, random_z_dim=10, reset_random_after_episode=False):
        super().__init__(config, seed, random_z_dim, reset_random_after_episode, single_env_name='hole_double')
        
class IphyreAdversarialHandDesignDownEnv(IphyreAdversarialHandDesignSingleEnv):
    def __init__(self, config={}, seed=None, random_z_dim=10, reset_random_after_episode=False):
        super().__init__(config, seed, random_z_dim, reset_random_after_episode, single_env_name='hole_down')
        
class IphyreAdversarialHandDesignHighBallEnv(IphyreAdversarialHandDesignSingleEnv):
    def __init__(self, config={}, seed=None, random_z_dim=10, reset_random_after_episode=False):
        super().__init__(config, seed, random_z_dim, reset_random_after_episode, single_env_name='hole_high_ball')

class IphyreAdversarialHandDesignHighGapEnv(IphyreAdversarialHandDesignSingleEnv):
    def __init__(self, config={}, seed=None, random_z_dim=10, reset_random_after_episode=False):
        super().__init__(config, seed, random_z_dim, reset_random_after_episode, single_env_name='hole_high_gap')

class IphyreAdversarialHandDesignOutsideEnv(IphyreAdversarialHandDesignSingleEnv):
    def __init__(self, config={}, seed=None, random_z_dim=10, reset_random_after_episode=False):
        super().__init__(config, seed, random_z_dim, reset_random_after_episode, single_env_name='hole_outside')

class IphyreAdversarialHandDesignReverseEnv(IphyreAdversarialHandDesignSingleEnv):
    def __init__(self, config={}, seed=None, random_z_dim=10, reset_random_after_episode=False):
        super().__init__(config, seed, random_z_dim, reset_random_after_episode, single_env_name='hole_reverse')

class IphyreAdversarialHandDesignRightEnv(IphyreAdversarialHandDesignSingleEnv):
    def __init__(self, config={}, seed=None, random_z_dim=10, reset_random_after_episode=False):
        super().__init__(config, seed, random_z_dim, reset_random_after_episode, single_env_name='hole_right')

class IphyreAdversarialHandDesignSkipGapEnv(IphyreAdversarialHandDesignSingleEnv):
    def __init__(self, config={}, seed=None, random_z_dim=10, reset_random_after_episode=False):
        super().__init__(config, seed, random_z_dim, reset_random_after_episode, single_env_name='hole_skip_gap')

class IphyreAdversarialHandDesignSkipGapHardEnv(IphyreAdversarialHandDesignSingleEnv):
    def __init__(self, config={}, seed=None, random_z_dim=10, reset_random_after_episode=False):
        super().__init__(config, seed, random_z_dim, reset_random_after_episode, single_env_name='hole_skip_gap_hard')

class IphyreAdversarialHandDesignSwapEnv(IphyreAdversarialHandDesignSingleEnv):
    def __init__(self, config={}, seed=None, random_z_dim=10, reset_random_after_episode=False):
        super().__init__(config, seed, random_z_dim, reset_random_after_episode, single_env_name='hole_swap')


if hasattr(__loader__, 'name'):  
  module_path = __loader__.name
elif hasattr(__loader__, 'fullname'):
  module_path = __loader__.fullname

gym_register(id='Iphyre-Adversarial-v0',
             entry_point=module_path + ':IphyreAdversarialEnv',
             max_episode_steps=2000)

gym_register(id='Iphyre-AdversarialVLM4k-v0',
             entry_point=module_path + ':IphyreAdversarialVLM4kEnv',
             max_episode_steps=2000)

gym_register(id='Iphyre-AdversarialVLM10k-v0',
             entry_point=module_path + ':IphyreAdversarialVLM10kEnv',
             max_episode_steps=2000)

gym_register(id='Iphyre-AdversarialClaudeVLM10k-v0',
             entry_point=module_path + ':IphyreAdversarialClaudeVLM10kEnv',
             max_episode_steps=2000)

gym_register(id='Iphyre-AdversarialGeminiVLM10k-v0',
             entry_point=module_path + ':IphyreAdversarialGeminiVLM10kEnv',
             max_episode_steps=2000)

gym_register(id='Iphyre-AdversarialProceduralRotate-v0',
             entry_point=module_path + ':IphyreAdversarialProceduralRotateEnv',
             max_episode_steps=2000)

gym_register(id='Iphyre-AdversarialProceduralShift-v0',
             entry_point=module_path + ':IphyreAdversarialProceduralShiftEnv',
             max_episode_steps=2000)

gym_register(id='Iphyre-Game-v0',
             entry_point=module_path + ':IphyreGameEnv',
             max_episode_steps=2000)

gym_register(id='IphyreAdversarialHandDesignBlockAngleEnv-v0',
             entry_point=module_path + ':IphyreAdversarialHandDesignBlockAngleEnv',
             max_episode_steps=2000)
gym_register(id='IphyreAdversarialHandDesignBlockAngleHardEnv-v0',
             entry_point=module_path + ':IphyreAdversarialHandDesignBlockAngleHardEnv',
             max_episode_steps=2000)
gym_register(id='IphyreAdversarialHandDesignBlockNoGapEnv-v0',
             entry_point=module_path + ':IphyreAdversarialHandDesignBlockNoGapEnv',
             max_episode_steps=2000)
gym_register(id='IphyreAdversarialHandDesignBlockOrderEnv-v0',
             entry_point=module_path + ':IphyreAdversarialHandDesignBlockOrderEnv',
             max_episode_steps=2000)
gym_register(id='IphyreAdversarialHandDesignConfuseEnv-v0',
             entry_point=module_path + ':IphyreAdversarialHandDesignConfuseEnv',
             max_episode_steps=2000)
gym_register(id='IphyreAdversarialHandDesignDoubleEnv-v0',
             entry_point=module_path + ':IphyreAdversarialHandDesignDoubleEnv',
             max_episode_steps=2000)
gym_register(id='IphyreAdversarialHandDesignDownEnv-v0',
             entry_point=module_path + ':IphyreAdversarialHandDesignDownEnv',
             max_episode_steps=2000)
gym_register(id='IphyreAdversarialHandDesignHighBallEnv-v0',
             entry_point=module_path + ':IphyreAdversarialHandDesignHighBallEnv',
             max_episode_steps=2000)
gym_register(id='IphyreAdversarialHandDesignHighGapEnv-v0',
             entry_point=module_path + ':IphyreAdversarialHandDesignHighGapEnv',
             max_episode_steps=2000)
gym_register(id='IphyreAdversarialHandDesignOutsideEnv-v0',
             entry_point=module_path + ':IphyreAdversarialHandDesignOutsideEnv',
             max_episode_steps=2000)
gym_register(id='IphyreAdversarialHandDesignReverseEnv-v0',
             entry_point=module_path + ':IphyreAdversarialHandDesignReverseEnv',
             max_episode_steps=2000)
gym_register(id='IphyreAdversarialHandDesignRightEnv-v0',
             entry_point=module_path + ':IphyreAdversarialHandDesignRightEnv',
             max_episode_steps=2000)
gym_register(id='IphyreAdversarialHandDesignSkipGapEnv-v0',
             entry_point=module_path + ':IphyreAdversarialHandDesignSkipGapEnv',
             max_episode_steps=2000)
gym_register(id='IphyreAdversarialHandDesignSkipGapHardEnv-v0',
             entry_point=module_path + ':IphyreAdversarialHandDesignSkipGapHardEnv',
             max_episode_steps=2000)
gym_register(id='IphyreAdversarialHandDesignSwapEnv-v0',
             entry_point=module_path + ':IphyreAdversarialHandDesignSwapEnv',
             max_episode_steps=2000)