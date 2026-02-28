from .spec import BenchmarkSpec

# ---------------------------------------------------------------------------
# Iphyre
# ---------------------------------------------------------------------------

_IPHYRE_ENV_CFG = dict(
    env_name="Iphyre-Adversarial-v0",
    num_processes=64,
    num_steps=256,
    gamma=0.99,
    gae_lambda=0.95,
    use_gae=True,
    handle_timelimits=True,
    # ued
    ued_algo="domain_randomization",
    use_plr=False,
    use_reset_random_dr=True,
    reset_random_after_episode=False,
    # SFL learnability (only used when ued_algo="sfl")
    update_learnability_every_iterations=10,
    learnability_c=0.0,
    learnability_buffer_size=1000,
    learnability_alpha=0.5,
    learnability_staleness=0.5,
    top_k_to_sample_uniformly=100,
    sfl_buffer_n_size=1000,
    sfl_buffer_nl_size=64,
)

_IPHYRE_MODEL_CFG = dict(
    recurrent_agent=False,
    recurrent_adversary_env=False,
    recurrent_arch="lstm",
    recurrent_hidden_size=1,
    should_freeze_embedding=False,
)

_IPHYRE_PPO_CFG = dict(
    algo="ppo",
    lr=2.5e-4,
    eps=1e-5,
    clip_param=0.2,
    clip_value_loss=False,
    ppo_epoch=30,
    num_mini_batch=4,
    value_loss_coef=0.5,
    entropy_coef=0.01,
    adv_entropy_coef=0.01,
    max_grad_norm=0.5,
    adv_max_grad_norm=0.5,
    kl_loss_coef=0.0,
    normalize_returns=False,
    use_popart=False,
    log_grad_norm=True,
    log_action_complexity=False,
)

_IPHYRE_EVAL_CFG = dict(
    test_env_names=(
        "Iphyre-HandDesign-v0,"
        "Iphyre-ProceduralRotate-v0,"
        "Iphyre-ProceduralShift-v0,"
        "Iphyre-VLMGeneratedRotate-v0,"
        "Iphyre-VLMGeneratedShift-v0"
    ),
    test_num_episodes=20,
    test_num_processes=1,
    test_interval=20,
)

IPHYRE_SPEC = BenchmarkSpec(
    env_name="Iphyre-Adversarial-v0",
    env_cfg=_IPHYRE_ENV_CFG,
    model_cfg=_IPHYRE_MODEL_CFG,
    ppo_cfg=_IPHYRE_PPO_CFG,
    eval_cfg=_IPHYRE_EVAL_CFG,
)
