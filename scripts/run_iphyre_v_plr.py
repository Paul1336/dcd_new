import argparse
import subprocess
import sys


PARAMS = {
    # Env
    'env_name':                             'Iphyre-AdversarialVLM4k-v0',
    'ued_algo':                             'plr',

    # Rollout / PPO
    'num_processes':                        64,
    'num_env_steps':                        30_000_000,
    'num_steps':                            256,
    'ppo_epoch':                            30,
    'num_mini_batch':                       4,
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

    # Checkpointing
    'checkpoint':                           True,
    'checkpoint_basis':                     'student_grad_updates',

    # PLR (level replay curriculum)
    'level_replay_strategy':                'value_l1',
    'level_replay_schedule':                'proportionate',
    'level_replay_score_transform':         'rank',
    'level_replay_temperature':             0.1,
    'level_replay_eps':                     0.05,
    'level_replay_rho':                     1.0,
    'level_replay_prob':                    0.95,
    'level_replay_alpha':                   1.0,
    'staleness_coef':                       0.3,
    'staleness_transform':                  'power',
    'staleness_temperature':                1.0,
    'train_full_distribution':              True,
    'level_replay_seed_buffer_size':        4000,
    'level_replay_seed_buffer_priority':    'replay_support',

    # Evaluation
    'test_env_names':                       'Iphyre-HandDesign-v0,Iphyre-HandDesign-v1,Iphyre-ProceduralRotate-v0,Iphyre-ProceduralShift-v0,Iphyre-VLMGeneratedRotate-v0,Iphyre-VLMGeneratedShift-v0',
    'test_interval':                        20,
    'test_num_episodes':                    20,
    'test_num_processes':                   1,

    # Screenshots
    'screenshot_interval':                  0,

    # Logging
    'log_interval':                         1,
    'log_plr_buffer_stats':                 True,
    'log_replay_complexity':                True,
    'log_action_complexity':                False,
    'log_grad_norm':                        True,
}

PROCEDURAL_OVERRIDES = {
    'env_name':         'Iphyre-Adversarial-v0',
}

EMBEDDING_OVERRIDES = {
    'obs_type':   'embedding',
    'clip_model': 'ViT-B/32',
}

TEST_OVERRIDES = {
    'num_processes':                        2,
    'num_env_steps':                        2048,
    'num_steps':                            128,
    'ppo_epoch':                            1,
    'num_mini_batch':                       1,
    'level_replay_seed_buffer_size':        20,
    'test_interval':                        2,
    'test_num_episodes':                    1,
    'vlm_env_max_tasks':                    20,
    'screenshot_interval':                  0,
}

BASE_SEED  = 88
NUM_TRIALS = 3


def build_cmd(seed, device, log_dir, method, exp_name,
              procedural=False, vlm_embedding=False, test=False, fake_clip=False,
              ball_relative=False):
    params = dict(PARAMS)
    if procedural:
        params.update(PROCEDURAL_OVERRIDES)
    if vlm_embedding:
        params.update(EMBEDDING_OVERRIDES)
        params['clip_device'] = 'cpu' if test else device
    if test:
        params.update(TEST_OVERRIDES)
    if fake_clip:
        params['fake_clip'] = True
    if ball_relative:
        params['use_ball_relative'] = True
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
    parser.add_argument('--device',        type=str, default='cuda:0')
    parser.add_argument('--log_dir',       type=str, default='~/logs/dcd/')
    parser.add_argument('--method',        type=str, default=None,
                        help='W&B method tag. Defaults to Iphyre-{V,VE,P}-PLR.')
    parser.add_argument('--exp_name',      type=str, default=None,
                        help='Experiment name. Defaults to iphyre_{v,ve,p}_plr.')
    parser.add_argument('--test',          action='store_true',
                        help='Smoke-test: tiny dataset + few steps, 1 trial, no confirm prompt.')
    parser.add_argument('--procedural',    action='store_true',
                        help='Use procedural env (Iphyre-Adversarial-v0) instead of VLM-generated levels.')
    parser.add_argument('--vlm_embedding', action='store_true',
                        help='Use frozen CLIP image embeddings instead of symbolic observations.')
    parser.add_argument('--fake_clip',     action='store_true',
                        help='Replace CLIP with zero embeddings for timing (requires --vlm_embedding).')
    parser.add_argument('--ball_relative', action='store_true',
                        help='Encode all positions relative to the ball center.')
    args = parser.parse_args()

    if args.procedural:
        variant = 'P'
    elif args.vlm_embedding:
        variant = 'VE'
    else:
        variant = 'V'
    if args.method is None:
        args.method = f'Iphyre-{variant}-PLR'
    if args.exp_name is None:
        args.exp_name = f'iphyre_{variant.lower()}_plr'

    num_trials = 1 if args.test else NUM_TRIALS

    cmds = [
        build_cmd(
            seed=BASE_SEED + i,
            device=args.device,
            log_dir=args.log_dir,
            method=args.method,
            exp_name=args.exp_name,
            procedural=args.procedural,
            vlm_embedding=args.vlm_embedding,
            test=args.test,
            fake_clip=args.fake_clip,
            ball_relative=args.ball_relative,
        )
        for i in range(num_trials)
    ]

    mode_parts = []
    if args.test:
        mode_parts.append('TEST')
    if args.procedural:
        mode_parts.append('Procedural')
    if args.vlm_embedding:
        mode_parts.append('VLM-Embedding')
    if args.fake_clip:
        mode_parts.append('fake-CLIP')
    if args.ball_relative:
        mode_parts.append('BallRelative')
    mode_label = f' [{", ".join(mode_parts)}]' if mode_parts else ''

    print(f'=== {num_trials} trial(s) (seeds {BASE_SEED}–{BASE_SEED + num_trials - 1}){mode_label} ===\n')
    for i, cmd in enumerate(cmds):
        print(f'[Trial {i}] seed={BASE_SEED + i}')
        print(cmd)
        print()

    if not args.test:
        input('Press Enter to start, Ctrl+C to cancel...')
    run_parallel(cmds)
