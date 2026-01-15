
from .registration import make as gym_make
from .wrapper import ParallelAdversarialVecEnv, VecNormalize, VecMonitor, VecPreprocessImageWrapper

def load_vlm_gen_tasks_solvable(env_name):
    pass

# util/create_parallel_env.py


def _create_iphyre_env(args):
    try:
        env_names = load_vlm_gen_tasks_solvable(args, args.env_name)
    except Exception as e:
        raise RuntimeError(
            f"[EnvInitError] Failed to load Iphyre tasks, env name: {args.env_name}"
        ) from e
    make_fns = [lambda: gym_make(args, env_names) for _ in range(args.num_processes)]
    try:
        venv = ParallelAdversarialVecEnv(make_fns, args=args, adversary=True)
    except Exception as e:
        raise RuntimeError(
            "[EnvInitError] Failed to create ParallelAdversarialVecEnv"
        ) from e
    try:
        venv = VecMonitor(venv=venv, filename=None, keep_buf=100)
    except Exception as e:
        raise RuntimeError(
            "[EnvWrapperError] Failed to apply VecMonitor"
        ) from e
    try:
        venv = VecNormalize(venv=venv, ob=False, ret=args.normalize_returns)
    except Exception as e:
        raise RuntimeError(
            "[EnvWrapperError] Failed to apply VecNormalize"
        ) from e
    try:
        venv = VecPreprocessImageWrapper(
            venv=venv,
            obs_key=None,
            transpose_order=[2, 0, 1],
            scale=None,
        )
    except Exception as e:
        raise RuntimeError(
            "[EnvWrapperError] Failed to apply VecPreprocessImageWrapper"
        ) from e
    return venv, venv


def create_parallel_env(args):
    if args.num_processes <= 0:
        raise ValueError(f"num_processes must be > 0, got {args.num_processes}")
    if args.env_name.startswith('Iphyre'):
        venv, ued_venv = _create_iphyre_env(args)
    else:
        raise ValueError(
            f"Unsupported env_name: {args.env_name}. "
        )
    if args.singleton_env: # False by default
        seeds = [args.seed]*args.num_processes
    else:
        seeds = [args.seed + i for i in range(args.num_processes)]
    venv.set_seed(seeds)

    return venv, ued_venv

