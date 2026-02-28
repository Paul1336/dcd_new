from .model.registry import build_model
from .agent import ACAgent
from storage.ppo_storage import RolloutStorage


def create_agent(args, env, name="agent"):
    actor_critic = build_model(
        env_name=args.env_name,
        env=env,
        agent_type=name,
        obs_type=getattr(args, "obs_type", "symbolic"),
        should_freeze_embedding=getattr(args, "should_freeze_embedding", False),
    )

    if args.algo == "ppo":
        return _create_ppo_agent(args, actor_critic, env, name)
    else:
        raise ValueError(f"Unsupported RL algorithm {args.algo}.")


def _create_ppo_agent(args, actor_critic, env, name):
    try:
        use_proper_time_limits = (
            hasattr(args, "handle_timelimits") and args.handle_timelimits
        )

        storage = RolloutStorage(
            model=actor_critic,
            num_steps=args.num_steps,
            num_processes=args.num_processes,
            observation_space=env.observation_space,
            action_space=env.action_space,
            recurrent_hidden_state_size=getattr(args, "recurrent_hidden_size", 1),
            recurrent_arch=getattr(args, "recurrent_arch", "rnn"),
            use_proper_time_limits=use_proper_time_limits,
            use_popart=getattr(args, "use_popart", False),
        )

        agent = ACAgent(
            actor_critic=actor_critic,
            storage=storage,
            clip_param=args.clip_param,
            ppo_epoch=args.ppo_epoch,
            num_mini_batch=args.num_mini_batch,
            value_loss_coef=args.value_loss_coef,
            entropy_coef=args.entropy_coef,
            kl_loss_coef=getattr(args, "kl_loss_coef", 0.0),
            lr=args.lr,
            eps=args.eps,
            max_grad_norm=args.max_grad_norm,
            clip_value_loss=args.clip_value_loss,
            log_grad_norm=getattr(args, "log_grad_norm", False),
        ).to(args.device)

        return agent

    except Exception as e:
        raise RuntimeError(
            f"[AgentAlgoInitError] Failed to initialize PPO agent '{name}'."
        ) from e
