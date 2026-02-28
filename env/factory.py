from .registration import make as gym_make
from .wrapper import ParallelAdversarialVecEnv, VecNormalize, VecMonitor, VecPreprocessImageWrapper, VecCLIPEmbeddingWrapper


_VLM_ENV_TASK_DIRS = {
    'Iphyre-AdversarialVLM4k-v0':        'vlm_4k_task_dir_list',
    'Iphyre-AdversarialVLM10k-v0':       'vlm_10k_task_dir_list',
    'Iphyre-AdversarialClaudeVLM10k-v0': 'vlm_10k_claude_task_dir_list',
    'Iphyre-AdversarialGeminiVLM10k-v0': 'vlm_10k_gemini_task_dir_list',
}


def _apply_obs_wrappers(venv, args):
    obs_type = getattr(args, "obs_type", "symbolic")
    if obs_type == "embedding":
        venv = VecCLIPEmbeddingWrapper(
            venv=venv,
            clip_model_name=getattr(args, "clip_model", "ViT-B/32"),
            clip_device=getattr(args, "clip_device", "cpu"),
        )
    return venv


def _create_iphyre_adversarial_env(args):
    import env.benchmark.iphyre.adversarial  # noqa: F401 — triggers gym registrations

    def make_env():
        return gym_make(args.env_name)

    make_fns = [make_env for _ in range(args.num_processes)]
    try:
        venv = ParallelAdversarialVecEnv(make_fns, adversary=False)
    except Exception as e:
        raise RuntimeError("[EnvInitError] Failed to create ParallelAdversarialVecEnv") from e

    venv = VecMonitor(venv=venv, filename=None, keep_buf=100)
    venv = VecNormalize(venv=venv, ob=False, ret=args.normalize_returns)
    venv = _apply_obs_wrappers(venv, args)
    return venv, venv


def _create_iphyre_vlm_env(args):
    from env.benchmark.iphyre.adversarial import load_vlm_gen_tasks_solvable
    import env.benchmark.iphyre.datapath as datapath

    task_dir_list = getattr(datapath, _VLM_ENV_TASK_DIRS[args.env_name])

    # Populate PARAS in the main process (needed for evaluate_parallel_envs in learnability update)
    # and get the full list of env_names for the SFL curriculum pool.
    env_names, _ = load_vlm_gen_tasks_solvable(task_dir_list, should_check_solvable=False)

    max_tasks = getattr(args, 'vlm_env_max_tasks', -1)
    if max_tasks > 0:
        env_names = env_names[:max_tasks]

    def make_env():
        return gym_make(args.env_name, env_names=env_names)

    make_fns = [make_env for _ in range(args.num_processes)]
    try:
        venv = ParallelAdversarialVecEnv(make_fns, adversary=False)
    except Exception as e:
        raise RuntimeError("[EnvInitError] Failed to create ParallelAdversarialVecEnv") from e

    venv = VecMonitor(venv=venv, filename=None, keep_buf=100)
    venv = VecNormalize(venv=venv, ob=False, ret=args.normalize_returns)
    venv = _apply_obs_wrappers(venv, args)
    return venv, venv


def create_parallel_env(args):
    if args.num_processes <= 0:
        raise ValueError(f"num_processes must be > 0, got {args.num_processes}")

    if args.env_name in _VLM_ENV_TASK_DIRS:
        venv, ued_venv = _create_iphyre_vlm_env(args)
    elif args.env_name.startswith('Iphyre'):
        venv, ued_venv = _create_iphyre_adversarial_env(args)
    else:
        raise ValueError(f"Unsupported env_name: {args.env_name}.")

    seeds = (
        [args.seed] * args.num_processes
        if args.singleton_env
        else [args.seed + i for i in range(args.num_processes)]
    )
    venv.set_seed(seeds)
    return venv, ued_venv
