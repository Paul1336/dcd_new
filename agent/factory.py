# util/create_agent.py

from .model import IphyreModel
from .algo import PPO, ACAgent
from .storage import RolloutStorage

def _create_PPO_agent(args, actor_critic_model, name):
    try:
        algo = PPO(args, actor_critic=actor_critic_model)
        storage = RolloutStorage(
            args=args,
            model=actor_critic_model,
            use_proper_time_limits=False,
        )
        agent = ACAgent(algo=algo, storage=storage).to(args.device)
        return agent
    except Exception as e:
        raise RuntimeError(
            f"[AgentAlgoInitError] Failed to initialize PPO agent '{name}'."
        ) from e

def create_agent(args, name, env):
    valid_names = {"agent", "adversary_agent", "adversary_env"}
    if name not in valid_names:
        raise ValueError(
            f"Invalid agent name '{name}'. Expected one of {sorted(valid_names)}."
        )
    model = None
    agent = None
    if args.env_name.startswith('Iphyre'):
        try:
            model = IphyreModel(args=args, env=env, name=name)
        except Exception as e:
            raise RuntimeError(
                f"[AgentModelInitError] Failed to create model for env='{args.env_name}', name='{name}'."
            ) from e
    else:
        raise ValueError(f'Unsupported environment {args.env_name}.')
    if args.algo == 'ppo':
        agent = _create_PPO_agent(args, model, name)
    else:
        raise ValueError(f'Unsupported RL algorithm {args.algo}.')

    return agent

