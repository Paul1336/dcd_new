# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sys
import os
import csv
import json
import argparse
import fnmatch
import re
from collections import defaultdict

import numpy as np
import torch
from baselines.common.vec_env import DummyVecEnv
from baselines.logger import HumanOutputFormat
from tqdm import tqdm

import os
import matplotlib as mpl
import matplotlib.pyplot as plt

from gym.wrappers.record_episode_statistics import RecordEpisodeStatistics

from envs.registration import make as gym_make
from envs.multigrid.maze import *
from envs.multigrid.crossing import *
from envs.multigrid.fourrooms import *
from envs.multigrid.mst_maze import *
from envs.box2d import *
from envs.bipedalwalker import *
from envs.wrappers import VecMonitor, VecPreprocessImageWrapper, ParallelAdversarialVecEnv, \
    MultiGridFullyObsWrapper, VecFrameStack, CarRacingWrapper
from util import DotDict, str2bool, make_agent, create_parallel_env, is_discrete_actions
from arguments import parser

from iphyre.simulator import PARAS

"""
Example usage:

python -m eval \
--env_name=MultiGrid-SixteenRooms-v0 \
--xpid=<xpid> \
--base_path="~/logs/dcd" \
--result_path="eval_results/"
--verbose
"""
def parse_args():
    parser = argparse.ArgumentParser(description='Eval')

    parser.add_argument(
        '--base_path',
        type=str,
        default='~/logs/dcd',
        help='Base path to experiment results directories.')
    parser.add_argument(
        '--xpid',
        type=str,
        default='latest',
        help='Experiment ID (result directory name) for evaluation.')
    parser.add_argument(
        '--prefix',
        type=str,
        default=None,
        help='Experiment ID prefix for evaluation (evaluate all matches).'
    )
    parser.add_argument(
        '--env_names',
        type=str,
        default='MultiGrid-Labyrinth-v0',
        help='CSV string of evaluation environments.')
    parser.add_argument(
        '--result_path',
        type=str,
        default='eval_results/',
        help='Relative path to evaluation results directory.')
    parser.add_argument(
        '--benchmark',
        type=str,
        default=None,
        choices=['maze', 'f1', 'bipedal', 'poetrose'],
        help="Name of benchmark for evaluation.")
    parser.add_argument(
        '--accumulator',
        type=str,
        default=None,
        help="Function for accumulating across multiple evaluation runs.")
    parser.add_argument(
        '--singleton_env',
        type=str2bool, nargs='?', const=True, default=False,
        help="When using a fixed env, whether the same environment should also be reused across workers.")
    parser.add_argument(
        '--seed', 
        type=int, 
        default=1, 
        help='Random seed.')
    parser.add_argument(
        '--max_seeds', 
        type=int, 
        default=None, 
        help='Maximum number of matched experiment IDs to evaluate.')
    parser.add_argument(
        '--num_processes',
        type=int,
        default=2,
        help='Number of CPU processes to use.')
    parser.add_argument(
        '--max_num_processes',
        type=int,
        default=10,
        help='Maximum number of CPU processes to use.')
    parser.add_argument(
        '--num_episodes',
        type=int,
        default=100,
        help='Number of evaluation episodes per xpid per environment.')
    parser.add_argument(
        '--model_tar',
        type=str,
        default='model',
        help='Name of .tar to evaluate.')
    parser.add_argument(
        '--model_name',
        type=str,
        default='agent',
        choices=['agent', 'adversary_agent'],
        help='Which agent to evaluate.')
    parser.add_argument(
        '--deterministic',
        type=str2bool, nargs='?', const=True, default=False,
        help="Evaluate policy greedily.")
    parser.add_argument(
        '--verbose',
        type=str2bool, nargs='?', const=True, default=False,
        help="Show logging messages in stdout")
    parser.add_argument(
        '--render',
        type=str2bool, nargs='?', const=True, default=False,
        help="Render environment in first evaluation process to screen.")
    parser.add_argument(
        '--record_video',
        type=str2bool, nargs='?', const=True, default=False,
        help="Record video of first environment evaluation process.")

    return parser.parse_args()



def evaluate_parallel_envs(env_names, env_task_configs, agent, num_episodes=20, device='cpu'):
    eval_seed = 100

    episodic_returns = dict()
    episodic_lengths = dict()
    for env_id in env_names:
        episodic_returns[env_id] = []
        episodic_lengths[env_id] = []

    def make_env(env_name, env_task_config):
        def thunk():
            env = gym_make('Iphyre-Game-v0', env_name=env_name, env_task_config=env_task_config)
            env = RecordEpisodeStatistics(env)

            return env

        return thunk

    with torch.no_grad():
        env_seed = eval_seed

        envs = gym.vector.SyncVectorEnv(
            [
                make_env(env_name, env_task_config)
                for env_name, env_task_config in zip(env_names, env_task_configs)
            ]
        )

        obs = envs.reset(
        )  # Beware of the seed; missing the seed could cause issue in reproducibility.
        envs.seed(env_seed)

        episodic_return = torch.zeros(len(env_names))
        episodic_length = torch.zeros(len(env_names))

        obs = torch.from_numpy(obs).to(dtype=torch.float32, device=device)
        recurrent_hidden_states = torch.zeros(
            1, agent.algo.actor_critic.recurrent_hidden_state_size, device=device)
        if agent.algo.actor_critic.is_recurrent and agent.algo.actor_critic.rnn.arch == 'lstm':
            recurrent_hidden_states = (recurrent_hidden_states, torch.zeros_like(recurrent_hidden_states))
        masks = torch.ones(1, 1, device=device)
    
        while True:
            _, action, _, recurrent_hidden_states = agent.act(
                obs, recurrent_hidden_states, masks, deterministic=False)

            action = action.cpu().numpy()
            obs, reward, done, infos = envs.step(action)

            # print('action: ', action, reward)

            episodic_length += 1
            episodic_return += reward

            obs = torch.from_numpy(obs).to(dtype=torch.float32, device=device)

            # print('done: ', done)
            # print('infos: ', infos)

            for i, env_id in enumerate(env_names):
                if done[i]:
                    if len(episodic_returns[env_id]) < num_episodes:
                        episodic_returns[env_id].append(episodic_return[i].item())
                        episodic_lengths[env_id].append(episodic_length[i].item())

                        # print(
                        #     f"Finished for {env_id}, reward: {episodic_return[i].item()}, length: {episodic_length[i].item()}"
                        # )

                    episodic_return[i] = 0
                    episodic_length[i] = 0

            num_finished_env = 0
            for i, env_id in enumerate(env_names):
                if len(episodic_returns[env_id]) >= num_episodes:
                    num_finished_env += 1

            if num_finished_env == len(env_names):
                break

    results = dict()
    for env_id in env_names:
        results[env_id] = dict()
        results[env_id]["success_rate"] = np.mean(
            np.array(episodic_returns[env_id]) > 0.9
        )
        # results[env_id]["mean_return"] = np.mean(episodic_returns[env_id])
        # results[env_id]["std_return"] = np.std(episodic_returns[env_id])
        # results[env_id]["mean_length"] = np.mean(episodic_lengths[env_id])
        # results[env_id]["std_length"] = np.std(episodic_lengths[env_id])
        # print(
        #     f"[{env_id}] Success rate: [{results[env_id]['success_rate']}] Return: [{results[env_id]['mean_return']}, {results[env_id]['std_return']}] Length: [{results[env_id]['mean_length']}, {results[env_id]['std_length']}]"
        # )

    return results



class Evaluator(object):
    def __init__(self, 
        test_suite_names, 
        num_processes, 
        num_episodes=10, 
        record_video=False, 
        device='cpu', 
        **kwargs):
        self.kwargs = kwargs # kwargs for env wrappers
        self.test_suite_names = test_suite_names
        self.device = device
        self.record_video = record_video
        self.num_processes = num_processes
        self.num_episodes = num_episodes
        self.kwargs = kwargs
        if 'Bipedal' in test_suite_names[0]:
            self.solved_threshold = 230
        elif 'Iphyre' in test_suite_names[0]:
            self.solved_threshold = 0.9
        else:
            self.solved_threshold = 0
        print('Solved threshold: ', self.solved_threshold)

        # self.venv_suite = { test_suite_name: [] for test_suite_name in self.test_suite_names }

        # for test_suite_name in self.test_suite_names:
        #     print('Preparing test suite: ', test_suite_name)

        #     if test_suite_name.startswith('Iphyre'):
        #         env_names, env_task_configs = self.load_test_suite_envs(test_suite_name)
        #     else:
        #         env_names = [test_suite_name]

        #     print('Env names: ', env_names)
        #     self._init_parallel_envs(
        #         env_names, self.num_processes, device=self.device, record_video=self.record_video, test_suite_name=test_suite_name, env_task_configs=env_task_configs, **self.kwargs)


    def get_stats_keys(self):
        keys = []
        # for env_name in self.env_names:
        #     keys += [f'solved_rate:{env_name}', f'test_returns:{env_name}']
        return keys

    @staticmethod
    def make_env(env_name, env_task_config=None, record_video=False, **kwargs):
        if kwargs.get('test_suite_name', '').startswith('Iphyre'):
            env = gym_make('Iphyre-Game-v0', env_name=env_name, env_task_config=env_task_config)
        else:
            if env_name in ['BipedalWalker-v3', 'BipedalWalkerHardcore-v3']:
                env = gym.make(env_name)
            else:
                env = gym_make(env_name)
            
        # print('env = ', env, type(env))
        # if env_name == 'Iphyre-HandDesign-v0':
        # 	print('env = ', env, type(env), env)
        # 	print(env.get_env_ids())

        is_multigrid = env_name.startswith('MultiGrid')
        is_car_racing = env_name.startswith('CarRacing')

        if is_car_racing:
            grayscale = kwargs.get('grayscale', False)
            num_action_repeat = kwargs.get('num_action_repeat', 8)
            nstack = kwargs.get('frame_stack', 4)
            crop = kwargs.get('crop_frame', False)

            env = CarRacingWrapper(
                env=env,
                grayscale=grayscale, 
                reward_shaping=False,
                num_action_repeat=num_action_repeat,
                nstack=nstack,
                crop=crop,
                eval_=True)

            if record_video:
                from gym.wrappers.monitor import Monitor
                env = Monitor(env, "videos/", force=True)
                print('Recording video!', flush=True)

        if is_multigrid and kwargs.get('use_global_policy'):
            env = MultiGridFullyObsWrapper(env, is_adversarial=False)

        return env

    @staticmethod
    def wrap_venv(venv, env_name, device='cpu'):
        is_multigrid = env_name.startswith('MultiGrid') or env_name.startswith('MiniGrid')
        is_car_racing = env_name.startswith('CarRacing')
        is_bipedal = env_name.startswith('BipedalWalker')
        is_iphyre = env_name.startswith('Iphyre')

        obs_key = None
        scale = None
        if is_multigrid:
            obs_key = 'image'
            scale = 10.0

        # Channels first
        transpose_order = None
        if is_car_racing or is_multigrid:
            transpose_order = [2,0,1]

        venv = VecMonitor(venv=venv, filename=None, keep_buf=100)

        venv = VecPreprocessImageWrapper(venv=venv, obs_key=obs_key,
                transpose_order=transpose_order, scale=scale, device=device)

        return venv

    def _init_parallel_envs(self, env_names, num_processes, device=None, record_video=False, **kwargs):
        self.env_names = env_names
        self.num_processes = num_processes
        self.device = device
        self.test_suite_name = kwargs.get('test_suite_name', '')
        self.venv_suite[self.test_suite_name] = { env_name: None for env_name in env_names}
        env_task_configs = kwargs.get('env_task_configs', [None]*len(env_names))

        make_fn = []
        for env_name, env_task_config in zip(env_names, env_task_configs):
            make_fn = [lambda: Evaluator.make_env(env_name, env_task_config, record_video, **kwargs)]*self.num_processes
            venv = ParallelAdversarialVecEnv(make_fn, adversary=False, is_eval=True)
            venv = Evaluator.wrap_venv(venv, env_name, device=device)
            self.venv_suite[self.test_suite_name][env_name] = venv

        self.is_discrete_actions = is_discrete_actions(self.venv_suite[self.test_suite_name][env_names[0]])

    def close(self):
        for _, venv in self.venv_suite[self.test_suite_name].items():
            venv.close()
    



    def load_test_suite_envs(self, test_suite_name):
        print('Load test suite: ', test_suite_name)

        task_suite_path = {
            "Iphyre-HandDesign-v0": "../iphyre/test_toy20250110/20250525/output_hand_test",
            "Iphyre-ProceduralShift-v0": "../iphyre/test_toy20250110/20250602/output_eval_shift",
            "Iphyre-ProceduralRotate-v0": "../iphyre/test_toy20250110/20250602/output_eval_rotate",
            "Iphyre-VLMGeneratedShift-v0": "../iphyre/test_toy20250110/20250427/output_shift",
            "Iphyre-VLMGeneratedRotate-v0": "../iphyre/test_toy20250110/20250427/output_rotate",
        }

        # To faciliate testing
        test_hard_limit = 100

        tasks_path = task_suite_path[test_suite_name]
        task_dirs = os.listdir(tasks_path)

        env_names = []
        env_task_configs = []

        for task_dir in task_dirs:
            config_path = os.path.join(tasks_path, task_dir, 'config.json')

            config = json.load(open(config_path))
            task_config = config['config']
            task_name = task_dir

            if "VLM" in test_suite_name:
                if config["success_rate"] > 0.0: # and task_name == "20250428_175035_game_15":
                    env_names.append(task_name)
                    env_task_configs.append(task_config)
            else:
                env_names.append(task_name)
                env_task_configs.append(task_config)

            if len(env_names) >= test_hard_limit:
                break

        return env_names, env_task_configs
        
    def evaluate(self, agent, deterministic=False, show_progress=False, render=False, accumulator='mean'):
        # Process each env_name individually, because each env_name has different env_ids in our evaluation setup.
        import time
        stats = {}
        for test_suite_name in self.test_suite_names:
            print('Running test suite: ', test_suite_name)
            start_time = time.time()

            env_names, env_task_configs = self.load_test_suite_envs(test_suite_name)
            eval_results = evaluate_parallel_envs(env_names, env_task_configs, agent, num_episodes=self.num_episodes, device=self.device)

            # eval_results = self.evaluate_inner(agent, deterministic, show_progress, render, accumulator, test_suite_name)
            # stats.update(eval_results)  

            total_success_rate = []

            for env_id in env_names:
                total_success_rate.append(eval_results[env_id]['success_rate'])
                print(
                    f"[{env_id}] Success rate: [{eval_results[env_id]['success_rate']}]"
                )
                stats[f'{test_suite_name}/{env_id}/success_rate'] = eval_results[env_id]['success_rate']

            print('Test suite stats: ', eval_results)
            print(f'Test suite {test_suite_name} average success rate: {np.mean(total_success_rate)}')
            stats[f'{test_suite_name}/mean_success_rate'] = np.mean(total_success_rate)
            print(f'Time taken: {time.time() - start_time} seconds')
        return stats

    def evaluate_inner(self, 
        agent, 
        deterministic=False, 
        show_progress=False,
        render=False,
        accumulator='mean',
        test_suite_name=None):
        # Evaluate agent for N episodes
        env_returns = {}
        env_solved_episodes = {}
        
        for env_name, venv in self.venv_suite[test_suite_name].items():
            returns = []
            solved_episodes = 0

            obs = venv.reset()
            recurrent_hidden_states = torch.zeros(
                self.num_processes, agent.algo.actor_critic.recurrent_hidden_state_size, device=self.device)
            if agent.algo.actor_critic.is_recurrent and agent.algo.actor_critic.rnn.arch == 'lstm':
                recurrent_hidden_states = (recurrent_hidden_states, torch.zeros_like(recurrent_hidden_states))
            masks = torch.ones(self.num_processes, 1, device=self.device)

            pbar = None
            if show_progress:
                pbar = tqdm(total=self.num_episodes)

            while len(returns) < self.num_episodes:
                # Sample actions
                with torch.no_grad():
                    _, action, _, recurrent_hidden_states = agent.act(
                        obs, recurrent_hidden_states, masks, deterministic=deterministic)

                # Observe reward and next obs
                action = action.cpu().numpy()
                if not self.is_discrete_actions:
                    action = agent.process_action(action)
                obs, reward, done, infos = venv.step(action)

                masks = torch.tensor(
                    [[0.0] if done_ else [1.0] for done_ in done],
                    dtype=torch.float32,
                    device=self.device)

                for i, info in enumerate(infos):
                    if 'episode' in info.keys():
                        returns.append(info['episode']['r'])
                        if returns[-1] > self.solved_threshold:
                            solved_episodes += 1
                        if pbar:
                            pbar.update(1)

                        # zero hidden states
                        if agent.is_recurrent:
                            recurrent_hidden_states[0][i].zero_()
                            recurrent_hidden_states[1][i].zero_()

                        if len(returns) >= self.num_episodes:
                            break

                if render:
                    venv.render_to_screen()

            if pbar:
                pbar.close()	

            print('Returns: ', returns)
            print('Solved episodes: ', solved_episodes)

            env_returns[env_name] = returns
            env_solved_episodes[env_name] = solved_episodes

        stats = {}

        all_success_rates = []

        for env_name, _ in self.venv_suite[test_suite_name].items():
            env_success_rate = env_solved_episodes[env_name]/self.num_episodes
            print(f'Evaluated env: {env_name} | Test returns {np.mean(env_returns[env_name])} | Success rate: {env_success_rate}')

            all_success_rates.append(env_success_rate)

            stats[f'{test_suite_name}/{env_name}/success_rate'] = env_success_rate
            stats[f'{test_suite_name}/{env_name}/mean_return'] = np.mean(env_returns[env_name])

        print(f'Test suite: {test_suite_name} | Average success rate: {np.mean(all_success_rates)}')
        stats[f'{test_suite_name}/mean_success_rate'] = np.mean(all_success_rates)
        
        return stats


def _get_f1_env_names():
    env_names = [f'CarRacingF1-{name}-v0' for name, cls in formula1.__dict__.items() if isinstance(cls, RaceTrack)]
    env_names.remove('CarRacingF1-LagunaSeca-v0')
    return env_names


def _get_zs_minigrid_env_names():
    env_names = [
        'MultiGrid-SixteenRooms-v0',
        'MultiGrid-SixteenRoomsFewerDoors-v0'
        'MultiGrid-Labyrinth-v0',
        'MultiGrid-Labyrinth2-v0',
        'MultiGrid-Maze-v0',
        'MultiGrid-Maze2-v0',
        "MultiGrid-LargeCorridor-v0",
        "MultiGrid-PerfectMazeMedium-v0",
        "MultiGrid-PerfectMazeLarge-v0",
        "MultiGrid-PerfectMazeXL-v0",
    ]
    return env_names


def _get_bipedal_env_names():
    env_names = [
        "BipedalWalker-v3",
        "BipedalWalkerHardcore-v3",
        "BipedalWalker-Med-Stairs-v0",
        "BipedalWalker-Med-PitGap-v0",
        "BipedalWalker-Med-StumpHeight-v0",
        "BipedalWalker-Med-Roughness-v0",
    ]
    return env_names


def _get_poet_rose_env_names():
    env_names = [f'BipedalWalker-POET-Rose-{id}-v0' for id in ['1a', '1b', '2a', '2b', '3a', '3b']]
    return env_names


if __name__ == '__main__':
    os.environ["OMP_NUM_THREADS"] = "1"

    display = None
    if sys.platform.startswith('linux'):
        print('Setting up virtual display')

        import pyvirtualdisplay
        display = pyvirtualdisplay.Display(visible=0, size=(1400, 900), color_depth=24)
        display.start()

    args = DotDict(vars(parse_args()))
    args.num_processes = min(args.num_processes, args.num_episodes)

    # === Determine device ====
    device = 'cpu'

    # === Load checkpoint ===
    # Load meta.json into flags object
    base_path = os.path.expandvars(os.path.expanduser(args.base_path))

    xpids = [args.xpid]
    if args.prefix is not None:
        all_xpids = fnmatch.filter(os.listdir(base_path), f"{args.prefix}*")
        filter_re = re.compile('.*_[0-9]*$')
        xpids = [x for x in all_xpids if filter_re.match(x)]

    # Set up results management
    os.makedirs(args.result_path, exist_ok=True)
    if args.prefix is not None:
        result_fname = args.prefix
    else:
        result_fname = args.xpid
    result_fname = f"{result_fname}-{args.model_tar}-{args.model_name}"
    result_fpath = os.path.join(args.result_path, result_fname)
    if os.path.exists(f'{result_fpath}.csv'):
        result_fpath = os.path.join(args.result_path, f'{result_fname}_redo')
    result_fpath = f'{result_fpath}.csv'

    csvout = open(result_fpath, 'w', newline='')
    csvwriter = csv.writer(csvout)

    env_results = defaultdict(list)

    # Get envs
    if args.benchmark == 'maze':
        env_names = _get_zs_minigrid_env_names()
    elif args.benchmark == 'f1':
        env_names = _get_f1_env_names()
    elif args.benchmark == 'bipedal':
        env_names = _get_bipedal_env_names()
    elif args.benchmark == 'poetrose':
        env_names = _get_poet_rose_env_names()
    else:
        env_names = args.env_names.split(',')

    num_envs = len(env_names)
    if num_envs*args.num_processes > args.max_num_processes:
        chunk_size = args.max_num_processes//args.num_processes
    else:
        chunk_size = num_envs

    num_chunks = int(np.ceil(num_envs/chunk_size))

    if args.record_video:
        num_chunks = 1
        chunk_size = 1
        args.num_processes = 1

    num_seeds = 0
    for xpid in xpids:
        if args.max_seeds is not None and num_seeds >= args.max_seeds:
            break

        xpid_dir = os.path.join(base_path, xpid)
        meta_json_path = os.path.join(xpid_dir, 'meta.json')

        model_tar = f'{args.model_tar}.tar'
        checkpoint_path = os.path.join(xpid_dir, model_tar)

        if os.path.exists(checkpoint_path):
            meta_json_file = open(meta_json_path)       
            xpid_flags = DotDict(json.load(meta_json_file)['args'])

            make_fn = [lambda: Evaluator.make_env(env_names[0])]
            dummy_venv = ParallelAdversarialVecEnv(make_fn, adversary=False, is_eval=True)
            dummy_venv = Evaluator.wrap_venv(dummy_venv, env_name=env_names[0], device=device)

            # Load the agent
            agent = make_agent(name='agent', env=dummy_venv, args=xpid_flags, device=device)

            try:
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
            except:
                continue
            model_name = args.model_name

            if 'runner_state_dict' in checkpoint:
                agent.algo.actor_critic.load_state_dict(checkpoint['runner_state_dict']['agent_state_dict'][model_name])
            else:
                agent.algo.actor_critic.load_state_dict(checkpoint)

            num_seeds += 1

            # Evaluate environment batch in increments of chunk size
            for i in range(num_chunks):
                start_idx = i*chunk_size
                env_names_ = env_names[start_idx:start_idx+chunk_size]

                # Evaluate the model
                xpid_flags.update(args)
                xpid_flags.update({"use_skip": False})

                evaluator = Evaluator(env_names_, 
                    num_processes=args.num_processes, 
                    num_episodes=args.num_episodes, 
                    frame_stack=xpid_flags.frame_stack,
                    grayscale=xpid_flags.grayscale,
                    use_global_critic=xpid_flags.use_global_critic,
                    record_video=args.record_video)

                stats = evaluator.evaluate(agent, 
                    deterministic=args.deterministic, 
                    show_progress=args.verbose,
                    render=args.render,
                    accumulator=args.accumulator)

                for k,v in stats.items():
                    if args.accumulator:
                        env_results[k].append(v)
                    else:
                        env_results[k] += v

                evaluator.close()
        else:
            print(f'No model path {checkpoint_path}')

    output_results = {}
    for k,_ in stats.items():
        results = env_results[k]
        output_results[k] = f'{np.mean(results):.2f} +/- {np.std(results):.2f}'
        q1 = np.percentile(results, 25, interpolation='midpoint')
        q3 = np.percentile(results, 75, interpolation='midpoint')
        median = np.median(results)
        output_results[f'iq_{k}'] = f'{q1:.2f}--{median:.2f}--{q3:.2f}'
        print(f"{k}: {output_results[k]}")
    HumanOutputFormat(sys.stdout).writekvs(output_results)

    if args.accumulator:
        csvwriter.writerow(['metric',] + [x for x in range(num_seeds)])
    else:
        csvwriter.writerow(['metric',] + [x for x in range(num_seeds*args.num_episodes)])
    for k,v in env_results.items():
        row = [k,] + v
        csvwriter.writerow(row)

    if display:
        display.stop()
