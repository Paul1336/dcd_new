# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import json
import sys
import os
import time
import timeit
import logging
from arguments import parser
import random
import torch
import gym
import matplotlib as mpl
import matplotlib.pyplot as plt
from baselines.logger import HumanOutputFormat

display = None

# if sys.platform.startswith('linux'):
#     print('Setting up virtual display')

#     import pyvirtualdisplay
#     display = pyvirtualdisplay.Display(visible=0, size=(1400, 900), color_depth=24)
#     display.start()

from envs.multigrid import *
from envs.multigrid.adversarial import *
from envs.box2d import *
from envs.bipedalwalker import *
from envs.iphyre import *
from envs.runners.adversarial_runner import AdversarialRunner 
from util import make_agent, FileWriter, safe_checkpoint, create_parallel_env, make_plr_args, save_images
from eval import Evaluator


if __name__ == '__main__':
    os.environ["OMP_NUM_THREADS"] = "1"

    args = parser.parse_args()

    suffix = str(random.randint(0, 1000))

    print("seeding", args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # torch.backends.cudnn.deterministic = args.torch_deterministic



    # === Configure logging ==
    if args.xpid is None:
        args.xpid = "lr-%s" % time.strftime("%Y%m%d-%H%M%S")
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    args.xpid = args.method + "_" + timestamp + "_" + suffix

    import wandb
    wandb.init(
        project="dcd",
        config=vars(args),
        name=args.xpid,
        monitor_gym=True,
        save_code=True,
    )
    

    log_dir = os.path.expandvars(os.path.expanduser(args.log_dir))
    filewriter = FileWriter(
        xpid=args.xpid, xp_args=args.__dict__, rootdir=log_dir
    )
    screenshot_dir = os.path.join(log_dir, args.xpid, 'screenshots')
    args.log_dir = os.path.join(log_dir, args.xpid)
    print('log_dir', log_dir)
    if not os.path.exists(screenshot_dir):
        os.makedirs(screenshot_dir, exist_ok=True)

    def log_stats(stats):
        filewriter.log(stats)
        if args.verbose:
            HumanOutputFormat(sys.stdout).writekvs(stats)

    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)
    else:
        logging.disable(logging.CRITICAL)

    # === Determine device ====
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = args.device
    if 'cuda' in device:
        torch.backends.cudnn.benchmark = True
        print('Using CUDA\n')

    # === Create parallel envs ===
    venv, ued_venv = create_parallel_env(args)

    is_training_env = args.ued_algo in ['paired', 'flexible_paired', 'minimax']
    is_paired = args.ued_algo in ['paired', 'flexible_paired']

    agent = make_agent(name='agent', env=venv, args=args, device=device)
    adversary_agent, adversary_env = None, None
    if is_paired or args.use_accel_paired:
        adversary_agent = make_agent(name='adversary_agent', env=venv, args=args, device=device)

    if is_training_env:
        adversary_env = make_agent(name='adversary_env', env=venv, args=args, device=device)
    if args.ued_algo == 'domain_randomization' and args.use_plr and not args.use_reset_random_dr:
        adversary_env = make_agent(name='adversary_env', env=venv, args=args, device=device)
        adversary_env.random()

    # === Create runner ===
    plr_args = None
    if args.use_plr:
        plr_args = make_plr_args(args, venv.observation_space, venv.action_space)
    train_runner = AdversarialRunner(
        args=args,
        venv=venv,
        agent=agent, 
        ued_venv=ued_venv, 
        adversary_agent=adversary_agent,
        adversary_env=adversary_env,
        flexible_protagonist=False,
        train=True,
        plr_args=plr_args,
        device=device)

    # === Configure checkpointing ===
    timer = timeit.default_timer
    initial_update_count = 0
    last_logged_update_at_restart = -1
    checkpoint_path = os.path.expandvars(
        os.path.expanduser("%s/%s/%s" % (log_dir, args.xpid, "model.tar"))
    )
    ## This is only used for the first iteration of finetuning
    if args.xpid_finetune:
        model_fname = f'{args.model_finetune}.tar'
        base_checkpoint_path = os.path.expandvars(
            os.path.expanduser("%s/%s/%s" % (log_dir, args.xpid_finetune, model_fname))
        )

    def checkpoint(index=None):
        if args.disable_checkpoint:
            return
        safe_checkpoint({'runner_state_dict': train_runner.state_dict()}, 
                        checkpoint_path,
                        index=index, 
                        archive_interval=args.archive_interval)
        logging.info("Saved checkpoint to %s", checkpoint_path)


    # === Load checkpoint ===
    # if args.checkpoint and os.path.exists(checkpoint_path):
    #     checkpoint_states = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    #     last_logged_update_at_restart = filewriter.latest_tick() # ticks are 0-indexed updates
    #     train_runner.load_state_dict(checkpoint_states['runner_state_dict'])
    #     initial_update_count = train_runner.num_updates
    #     logging.info(f"Resuming preempted job after {initial_update_count} updates\n") # 0-indexed next update
    # elif args.xpid_finetune and not os.path.exists(checkpoint_path):
    #     checkpoint_states = torch.load(base_checkpoint_path)
    #     state_dict = checkpoint_states['runner_state_dict']
    #     agent_state_dict = state_dict.get('agent_state_dict')
    #     optimizer_state_dict = state_dict.get('optimizer_state_dict')
    #     train_runner.agents['agent'].algo.actor_critic.load_state_dict(agent_state_dict['agent'])
    #     train_runner.agents['agent'].algo.optimizer.load_state_dict(optimizer_state_dict['agent'])

    if args.model_path:
        state_dict = torch.load(args.model_path, map_location=device)
        # state_dict = checkpoint_states['runner_state_dict']
        agent_state_dict = state_dict.get('agent_state_dict')
        optimizer_state_dict = state_dict.get('optimizer_state_dict')
        train_runner.agents['agent'].algo.actor_critic.load_state_dict(agent_state_dict['agent'])
        train_runner.agents['agent'].algo.optimizer.load_state_dict(optimizer_state_dict['agent'])
    
        print('Loaded model from ', args.model_path)

        if args.value_model_path:
            state_dict = torch.load(args.value_model_path, map_location=device)
            agent_state_dict = state_dict.get('agent_state_dict')

            critic_state_dict = train_runner.agents['agent'].algo.actor_critic.critic.state_dict()
            # Reset the critic
            for critic_key, critic_param in critic_state_dict.items():
                critic_state_dict[critic_key] = torch.zeros_like(critic_param)
            for key, param in agent_state_dict['agent'].items():
                if 'critic' in key:
                    for critic_key, critic_param in critic_state_dict.items():
                        if critic_key in key:
                            critic_state_dict[critic_key] = param
                            print('loaded critic param', key, 'to', critic_key)
            
            print('critic_state_dict', critic_state_dict)
            train_runner.agents['agent'].algo.actor_critic.critic.load_state_dict(critic_state_dict)
            print('Loaded value model from ', args.value_model_path)
    
    print('checkpoint_path', checkpoint_path)
    print('archive_interval', args.archive_interval)
    print('screenshot_interval', args.screenshot_interval)

    # === Set up Evaluator ===
    evaluator = None
    if args.test_env_names:
        evaluator = Evaluator(
            args.test_env_names.split(','), 
            num_processes=args.test_num_processes, 
            num_episodes=args.test_num_episodes,
            frame_stack=args.frame_stack,
            grayscale=args.grayscale,
            num_action_repeat=args.num_action_repeat,
            use_global_critic=args.use_global_critic,
            use_global_policy=args.use_global_policy,
            device=device)

    # === Train === 
    last_checkpoint_idx = getattr(train_runner, args.checkpoint_basis)
    update_start_time = timer()
    num_updates = int(args.num_env_steps) // args.num_steps // args.num_processes

    # Zero shot evaluation
    if not os.environ.get('LOCAL_TEST', False) and evaluator is not None:
        test_stats = evaluator.evaluate(train_runner.agents['agent'])
        test_stats['total_student_grad_updates'] = 0
        test_stats['global_step'] = 0
        wandb.log(test_stats)

    for j in range(1, num_updates + 1):

        if j >= args.early_stop_at_iteration and args.early_stop_at_iteration > 0:
            print(f"Early stopping at iteration {j}")
            break
        
        global_step = j * args.num_steps * args.num_processes
        print(f"Iteration {j} of {num_updates} global_step: {global_step}")
        stats = train_runner.run(global_step=global_step, iteration=j, total_iterations=num_updates)

        stats['global_step'] = global_step

        # === Perform logging ===
        if train_runner.num_updates <= last_logged_update_at_restart:
            continue

        log = (j % args.log_interval == 0 and j != 0) or j == num_updates - 1
        save_screenshot = \
            args.screenshot_interval > 0 and \
                (j % args.screenshot_interval == 0)

        print('train stats', stats)

        if log:
            # Eval
            print('Evaluating...')
            test_stats = {}
            if evaluator is not None and (j % args.test_interval == 0 or j == num_updates - 1):
                test_stats = evaluator.evaluate(train_runner.agents['agent'])
                stats.update(test_stats)
                if args.use_accel_paired:
                    adv_test_stats = evaluator.evaluate(train_runner.agents['adversary_agent'])
                    curr_keys = list(adv_test_stats.keys())
                    for curr_key in curr_keys:
                        adv_test_stats[f"advagent_{curr_key}"] = adv_test_stats[curr_key]
                        adv_test_stats.pop(curr_key, None)
                    stats.update(adv_test_stats)
            else:
                if evaluator is not None:
                    stats.update({k:None for k in evaluator.get_stats_keys()})

            update_end_time = timer()
            num_incremental_updates = 1 if j == 0 else args.log_interval
            sps = num_incremental_updates*(args.num_processes * args.num_steps) / (update_end_time - update_start_time)
            update_start_time = update_end_time

            print(f"test_stats: {test_stats}")

            # Step per second
            # For example, 600 sps
            # total 3M steps => 3000000/600/60/24 => 3.47 days
            stats.update({'sps': sps})
            stats.update(test_stats) # Ensures sps column is always before test stats
            log_stats(stats)

        wandb.log(stats)

        checkpoint_idx = getattr(train_runner, args.checkpoint_basis)

        os.makedirs(f"{args.log_dir}/checkpoints", exist_ok=True)
        if j % args.archive_interval == 0:
            torch.save(train_runner.state_dict(), f"{args.log_dir}/checkpoints/checkpoint_{global_step}.pt")
            print(f"Archived checkpoint after update {j}")

        # if checkpoint_idx != last_checkpoint_idx:
        #     is_last_update = j == num_updates - 1
        #     if is_last_update or \
        #         (train_runner.num_updates > 0 and checkpoint_idx % args.checkpoint_interval == 0):
        #         checkpoint(checkpoint_idx)
        #         logging.info(f"\nSaved checkpoint after update {j}")
        #         logging.info(f"\nLast update: {is_last_update}")
        #     elif train_runner.num_updates > 0 and args.archive_interval > 0 \
        #         and checkpoint_idx % args.archive_interval == 0:
        #         checkpoint(checkpoint_idx)
        #         logging.info(f"\nArchived checkpoint after update {j}")

        if save_screenshot:
            print('Saving screenshot...')
            level_info = train_runner.sampled_level_info
            if args.env_name.startswith('BipedalWalker'):
                encodings = venv.get_level()
                df = bipedalwalker_df_from_encodings(args.env_name, encodings)
                if args.use_editor and level_info:
                    df.to_csv(os.path.join(
                        screenshot_dir, 
                        f"update{j}-replay{level_info['level_replay']}-n_edits{level_info['num_edits'][0]}.csv"))
                else:
                    df.to_csv(os.path.join(
                        screenshot_dir, 
                        f'update{j}.csv'))
            else:
                venv.reset_agent()
                try:
                    full_obs = venv.remote_attr('cur_full_obs')
                    json.dump(full_obs, open(os.path.join(screenshot_dir, f'update{global_step}_obs.json'), 'w'))
                except:
                    print('No full obs')
            
                images = venv.get_images()
                if args.use_editor and level_info:
                    save_images(
                        images[:args.screenshot_batch_size], 
                        os.path.join(
                            screenshot_dir, 
                            f"update{global_step}-replay{level_info['level_replay']}-n_edits{level_info['num_edits'][0]}.png"), 
                        normalize=True, channels_first=False)
                else:
                    save_images(
                        images[:args.screenshot_batch_size], 
                        os.path.join(screenshot_dir, f'update{global_step}.png'),
                        normalize=True, channels_first=False)
                plt.close()

    # evaluator.close()
    venv.close()

    if display:
        display.stop()