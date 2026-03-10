import argparse
import subprocess
import sys


PARAMS = {
    # Env
    'env_name':                             'MultiGrid-AdversarialVLM-v0',
    'ued_algo':                             'sfl',

    # Rollout / PPO
    'num_processes':                        32,
    'num_env_steps':                        30_000_000,
    'num_steps':                            256,
    'ppo_epoch':                            5,
    'num_mini_batch':                       1,
    'lr':                                   1e-4,
    'max_grad_norm':                        0.5,
    'gamma':                                0.995,
    'gae_lambda':                           0.95,
    'value_loss_coef':                      0.5,
    'entropy_coef':                         0.01,
    'adv_entropy_coef':                     0.01,
    'clip_value_loss':                      False,
    'clip_param':                           0.2,
    'normalize_returns':                    False,

    # Architecture
    'recurrent_agent':                      True,
    'recurrent_adversary_env':              False,
    'recurrent_hidden_size':                256,

    # Env-specific
    'reward_shaping':                       True,
    'use_categorical_adv':                  True,
    'use_skip':                             False,
    'choose_start_pos':                     False,
    'sparse_rewards':                       False,
    'handle_timelimits':                    True,

    # Checkpointing
    'checkpoint':                           True,
    'checkpoint_basis':                     'student_grad_updates',

    # Learnability (SFL)
    'update_learnability_every_iterations': 10,
    'learnability_c':                       0.0,
    'learnability_buffer_size':             1000,
    'learnability_alpha':                   0.5,
    'top_k_to_sample_uniformly':            -1,
    'learnability_staleness':               0.1,

    # Evaluation
    'test_env_names':                       '',
    'test_interval':                        20,
    'test_num_episodes':                    10,
    'test_num_processes':                   1,

    # Screenshots / logging
    'screenshot_interval':                  0,
    'log_interval':                         1,
    'log_plr_buffer_stats':                 True,
    'log_replay_complexity':                True,
    'log_action_complexity':                False,
    'log_grad_norm':                        True,
}

PROCEDURAL_OVERRIDES = {
    'env_name':                     'MultiGrid-GoalLastAdversarial-v0',
    'learnability_staleness':       0.5,
    'top_k_to_sample_uniformly':    100,
}

TEST_OVERRIDES = {
    'num_processes':                        2,
    'num_env_steps':                        2048,
    'num_steps':                            128,
    'ppo_epoch':                            1,
    'num_mini_batch':                       1,
    'learnability_buffer_size':             10,
    'update_learnability_every_iterations': 2,
    'test_interval':                        2,
    'test_num_episodes':                    1,
    'vlm_env_max_tasks':                    20,
    'screenshot_interval':                  0,
}

BASE_SEED  = 88
NUM_TRIALS = 3


def build_cmd(seed, device, log_dir, method, exp_name, wandb_group=None, procedural=False, test=False):
    params = dict(PARAMS)
    if procedural:
        params.update(PROCEDURAL_OVERRIDES)
    if test:
        params.update(TEST_OVERRIDES)
    cmd = ['python', 'train.py']
    for k, v in params.items():
        cmd.append(f'--{k}={v}')
    cmd += [
        f'--seed={seed}',
        f'--device={device}',
        f'--log_dir={log_dir}',
        f'--method={method}',
        f'--exp_name={exp_name}',
    ]
    if wandb_group:
        cmd.append(f'--wandb_group={wandb_group}')
    return ' '.join(str(x) for x in cmd)


def run_parallel(cmds):
    procs = []
    try:
        for i, cmd in enumerate(cmds):
            print('-' * 80)
            print(f'[Trial {i}] {cmd}')
            print('-' * 80)
            procs.append(subprocess.Popen(cmd, shell=True, text=True))
        for i, p in enumerate(procs):
            ret = p.wait()
            if ret != 0:
                print(f'[Trial {i}] failed with exit code {ret}')
            else:
                print(f'[Trial {i}] completed successfully')
    except KeyboardInterrupt:
        print('\nStopping all trials...')
        for p in procs:
            p.terminate()
        sys.exit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device',     type=str, default='cuda:0')
    parser.add_argument('--log_dir',    type=str, default='~/logs/dcd/')
    parser.add_argument('--method',     type=str, default=None)
    parser.add_argument('--exp_name',   type=str, default=None)
    parser.add_argument('--test',       action='store_true')
    parser.add_argument('--procedural',   action='store_true',
                        help='Use procedural env instead of VLM task pool.')
    parser.add_argument('--wandb_group',  type=str, default=None,
                        help='W&B group name for run aggregation.')
    args = parser.parse_args()

    variant = 'P' if args.procedural else 'V'
    if args.method is None:
        args.method = f'MultiGrid-{variant}-SFL'
    if args.exp_name is None:
        args.exp_name = f'minigrid_{variant.lower()}_sfl'

    num_trials = 1 if args.test else NUM_TRIALS
    cmds = [
        build_cmd(seed=BASE_SEED + i, device=args.device, log_dir=args.log_dir,
                  method=args.method, exp_name=args.exp_name, wandb_group=args.wandb_group,
                  procedural=args.procedural, test=args.test)
        for i in range(num_trials)
    ]

    mode_parts = ['TEST'] if args.test else []
    if args.procedural:
        mode_parts.append('Procedural')
    mode_label = f' [{", ".join(mode_parts)}]' if mode_parts else ''

    print(f'=== {num_trials} trial(s) (seeds {BASE_SEED}–{BASE_SEED + num_trials - 1}){mode_label} ===\n')
    for i, cmd in enumerate(cmds):
        print(f'[Trial {i}] seed={BASE_SEED + i}')
        print(cmd)
        print()

    if not args.test:
        input('Press Enter to start, Ctrl+C to cancel...')
    run_parallel(cmds)
