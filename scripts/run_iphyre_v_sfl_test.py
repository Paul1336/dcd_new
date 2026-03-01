"""
Quick smoke-test for the iphyre-v-sfl pipeline.

Loads only 20 VLM tasks, uses 2 parallel envs, and stops after ~16 training
iterations so the full end-to-end flow (env creation → rollout → PPO update →
learnability → eval) can be verified in a few minutes on any machine.

Usage:
    cd dcd_new
    python scripts/run_iphyre_v_sfl_test.py [--device cuda:0] [--log_dir ~/logs/dcd/]
"""

import argparse
import subprocess
import sys


PARAMS = {
    # Env
    'env_name':                             'Iphyre-AdversarialVLM4k-v0',
    'ued_algo':                             'sfl',

    # Rollout / PPO  (reduced for smoke-test)
    'num_processes':                        2,
    'num_env_steps':                        2048,
    'num_steps':                            128,
    'ppo_epoch':                            1,
    'num_mini_batch':                       1,
    'lr':                                   2.5e-4,
    'max_grad_norm':                        0.5,
    'gamma':                                0.99,
    'gae_lambda':                           0.95,
    'value_loss_coef':                      0.5,
    'entropy_coef':                         0.01,
    'adv_entropy_coef':                     0.01,
    'clip_value_loss':                      False,
    'clip_param':                           0.2,
    'normalize_returns':                    False,

    # Architecture
    'recurrent_agent':                      False,
    'recurrent_adversary_env':              False,
    'recurrent_hidden_size':                1,

    # Env-specific
    'reward_shaping':                       True,
    'use_categorical_adv':                  True,
    'use_skip':                             False,
    'choose_start_pos':                     False,
    'sparse_rewards':                       False,
    'handle_timelimits':                    True,

    # Dataset cap  (reduced for smoke-test)
    'vlm_env_max_tasks':                    20,

    # Checkpointing
    'checkpoint':                           True,
    'checkpoint_basis':                     'student_grad_updates',

    # Learnability (SFL)  (reduced for smoke-test)
    'update_learnability_every_iterations': 2,
    'learnability_c':                       0.0,
    'learnability_buffer_size':             10,
    'learnability_alpha':                   0.5,
    'top_k_to_sample_uniformly':            -1,
    'learnability_staleness':               0.1,

    # Evaluation  (reduced for smoke-test)
    'test_env_names':                       'Iphyre-HandDesign-v0,Iphyre-ProceduralRotate-v0,Iphyre-ProceduralShift-v0,Iphyre-VLMGeneratedRotate-v0,Iphyre-VLMGeneratedShift-v0',
    'test_interval':                        2,
    'test_num_episodes':                    1,
    'test_num_processes':                   1,

    # Screenshots  (disabled: pygame.init() in subprocesses blocks on headless servers)
    'screenshot_interval':                  0,

    # Logging
    'log_interval':                         1,
    'log_plr_buffer_stats':                 True,
    'log_replay_complexity':                True,
    'log_action_complexity':                False,
    'log_grad_norm':                        True,
}

SEED = 88


def build_cmd(device, log_dir, method, exp_name):
    cmd = ['python', 'train.py']
    for k, v in PARAMS.items():
        cmd.append(f'--{k}={v}')
    cmd += [
        f'--seed={SEED}',
        f'--device={device}',
        f'--log_dir={log_dir}',
        f'--method={method}',
        f'--exp_name={exp_name}',
    ]
    return ' '.join(str(x) for x in cmd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device',   type=str, default='cuda:0')
    parser.add_argument('--log_dir',  type=str, default='~/logs/dcd/')
    parser.add_argument('--method',   type=str, default='Iphyre-V-SFL-test')
    parser.add_argument('--exp_name', type=str, default='iphyre_v_sfl_test')
    args = parser.parse_args()

    cmd = build_cmd(args.device, args.log_dir, args.method, args.exp_name)

    print('=== smoke-test (1 trial, seed=88, 20 tasks, ~16 iters) ===\n')
    print(cmd)
    print()

    try:
        ret = subprocess.run(cmd, shell=True, text=True).returncode
        if ret != 0:
            print(f'[FAILED] exit code {ret}')
            sys.exit(ret)
        print('[OK] smoke-test passed')
    except KeyboardInterrupt:
        print('\nCancelled.')
        sys.exit(1)
