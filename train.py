# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# === Python std libraries ===
import json
import sys
import os
import time
import timeit 
import logging 
import random 


# === Third-party libraries ===
import torch 
from torchvision import utils as vutils
import matplotlib.pyplot as plt 
import numpy as np
from baselines.logger import HumanOutputFormat 
import wandb


# === modules ===
from util.FileWriter import FileWriter
from util.Parser import Parser
from util import make_plr_args, save_images
from env import create_parallel_env
from agent import create_agent
from eval import create_evaluator
from runner import create_runner

def train(args):
    os.environ["OMP_NUM_THREADS"] = "1"

    # === Set random seed ===
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    print("seeding", args.seed)

    # === Determine device ====
    if args.device.startswith("cuda"):
        if not torch.cuda.is_available():
            print("CUDA not available, using CPU")
            args.device = "cpu"
        else:
            try:
                torch.cuda.get_device_properties(torch.device(args.device))
                torch.backends.cudnn.benchmark = True
                print(f"Using CUDA device: {args.device}\n")
            except RuntimeError as e:
                logging.exception(f"[DeviceConfigError]")
                print("CUDA not available, using CPU")
                args.device = "cpu"

    # === Configure logging ==
    suffix = str(random.randint(0, 1000))
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    args.xpid = args.method + "_" + timestamp + "_" + suffix

    log_dir = os.path.expandvars(os.path.expanduser(args.log_dir))
    try:
        filewriter = FileWriter(xpid=args.xpid, xp_args=args.__dict__, rootdir=log_dir)
    except Exception :
        logging.exception(f"[FileWriterError]")
        raise
    args.screenshot_dir = os.path.join(log_dir, args.xpid, 'screenshots')
    args.log_dir = os.path.join(log_dir, args.xpid)
    print('log_dir', args.log_dir)

    # === Create parallel envs ===
    try:
        venv, ued_venv = create_parallel_env(args)
    except Exception :
        logging.exception(f"[EnvCreationError]")
        raise
    args.is_training_env = args.ued_algo in ['paired', 'flexible_paired', 'minimax']
    args.is_paired = args.ued_algo in ['paired', 'flexible_paired']

    # === Create agents ===
    try:
        agent = create_agent(args=args, name='agent', env=venv)
        adversary_agent, adversary_env = None, None
        if args.is_paired or args.use_accel_paired:
            adversary_agent = create_agent(args=args, name='adversary_agent', env=venv)
        if args.is_training_env:
            adversary_env = create_agent(args=args, name='adversary_env', env=venv)
        if args.ued_algo == 'domain_randomization' and args.use_plr and not args.use_reset_random_dr:
            adversary_env = create_agent(args=args, name='adversary_env', env=venv)
            adversary_env.random()
    except Exception :
        logging.exception(f"[AgentCreationError]")
        raise

    # === Create runner ===
    try:
        plr_args = None
        if args.use_plr:
            plr_args = make_plr_args(args, venv.observation_space, venv.action_space)
        runner = create_runner(
            args=args,
            venv=venv,
            agent=agent, 
            ued_venv=ued_venv, 
            adversary_agent=adversary_agent,
            adversary_env=adversary_env,
            train=True,
            plr_args=plr_args,
            flexible_protagonist=False,)
    except Exception :
        logging.exception(f"[RunnerCreationError]")
        raise

    # === Create evaluator ===
    try:
        evaluator = None
        if args.test_env_names:
            evaluator = create_evaluator(args=args)
    except Exception :
        logging.exception(f"[EvaluatorCreationError]")
        raise

    # === Initialize wandb ===
    try:
        wandb.init(
            project="dcd_new",
            config=vars(args),
            name=args.xpid,
            monitor_gym=True,
            save_code=True,
            tags=getattr(args, 'wandb_tags', None),
            group=getattr(args, 'wandb_group', None),
        )
    except Exception :
        logging.exception(f"[WandbInitError]")
        raise

    # === Train === 
    timer = timeit.default_timer
    update_start_time = timer()
    num_updates = int(args.num_env_steps) // args.num_steps // args.num_processes
    print('archive_interval', args.archive_interval)
    print('screenshot_interval', args.screenshot_interval)

    # Zero shot evaluation
    try:
        if not os.environ.get('LOCAL_TEST', False) and evaluator is not None:
            print('Performing zero-shot evaluation...')
            test_stats = evaluator.evaluate(runner.agents['agent'])
            test_stats['total_student_grad_updates'] = 0
            test_stats['global_step'] = 0
            wandb.log(test_stats)
    except Exception :
        logging.exception(f"[ZeroShotEvalError]")
        raise

    for j in range(1, num_updates + 1):

        if j >= args.early_stop_at_iteration and args.early_stop_at_iteration > 0:
            print(f"Early stopping at iteration {j}")
            break
        
        # === Run training ===
        global_step = j * args.num_steps * args.num_processes
        print(f"Iteration {j} of {num_updates} global_step: {global_step}")
        try:
            stats = runner.run(global_step=global_step, iteration=j, total_iterations=num_updates)
        except Exception :
            logging.exception(f"[RunnerRunError] In iteration {j},")
            raise

        # === Perform logging ===
        stats['global_step'] = global_step
        log = (j % args.log_interval == 0 and j != 0) or j == num_updates
        save_screenshot = (args.screenshot_interval > 0) and ((j % args.screenshot_interval == 0 or j == num_updates))
        print('train stats', stats)
        if log:
            # Eval
            print('Evaluating...')
            test_stats = {}
            if evaluator is not None and (j % args.test_interval == 0 or j == num_updates):
                test_stats = evaluator.evaluate(runner.agents['agent'])
                stats.update(test_stats)
                if args.use_accel_paired:
                    adv_test_stats = evaluator.evaluate(runner.agents['adversary_agent'])
                    curr_keys = list(adv_test_stats.keys())
                    for curr_key in curr_keys:
                        adv_test_stats[f"advagent_{curr_key}"] = adv_test_stats[curr_key]
                        adv_test_stats.pop(curr_key, None)
                    stats.update(adv_test_stats)
            else:
                if evaluator is not None:
                    stats.update({k:None for k in evaluator.get_stats_keys()})

            update_end_time = timer()
            num_incremental_updates = args.log_interval
            sps = num_incremental_updates*(args.num_processes * args.num_steps) / (update_end_time - update_start_time)
            update_start_time = update_end_time
            print(f"test_stats: {test_stats}")
            # Step per second
            # For example, 600 sps
            # total 3M steps => 3000000/600/60/24 => 3.47 days
            stats.update({'sps': sps})
            stats.update(test_stats) # Ensures sps column is always before test stats
            filewriter.log(stats)
            if args.verbose:
                HumanOutputFormat(sys.stdout).writekvs(stats)
        wandb.log(stats)

        try:
            os.makedirs(f"{args.log_dir}/checkpoints", exist_ok=True)
            # === Checkpointing ===
            if j % args.archive_interval == 0:
                torch.save(runner.state_dict(), f"{args.log_dir}/checkpoints/checkpoint_{global_step}.pt")
                print(f"Archived checkpoint after update {j}")
        except Exception :
            logging.exception(f"[CheckpointError]")

        # === Save screenshots ===
        if save_screenshot:
            try:
                os.makedirs(args.screenshot_dir, exist_ok=True)
                print('Saving screenshot...')
                level_info = runner.sampled_level_info
                venv.reset_agent()
                try:
                    full_obs = venv.remote_attr('cur_full_obs')
                    json.dump(full_obs, open(os.path.join(args.screenshot_dir, f'update{global_step}_obs.json'), 'w'))
                except Exception :
                    logging.exception(f'No full obs,')
                images = venv.get_images()
                if args.use_editor and level_info:
                    save_images(
                        images[:args.screenshot_batch_size], 
                        os.path.join(
                            args.screenshot_dir, 
                            f"update{global_step}-replay{level_info['level_replay']}-n_edits{level_info['num_edits'][0]}.png"), 
                        normalize=True, channels_first=False)
                else:
                    save_images(
                        images[:args.screenshot_batch_size], 
                        os.path.join(args.screenshot_dir, f'update{global_step}.png'),
                        normalize=True, channels_first=False)
                plt.close()
            except Exception :
                logging.exception(f"[ScreenshotError]")

    try:
        print('Closing environments and wandb...')
        venv.close()
        wandb.finish()
        if evaluator is not None:
            evaluator.close()
    except Exception :
        logging.exception(f"[ClosingError]")
        raise
   

if __name__ == '__main__':
    parser = Parser()
    args = parser.parse_args()
    train(args)