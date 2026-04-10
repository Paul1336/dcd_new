from .iphyre import IphyreNetwork
from .multigrid import MultigridNetwork


def model_for_iphyre_agent(env, agent_type="agent", obs_type="symbolic",
                           should_freeze_embedding=False, use_ball_relative=False):
    if "adversary_env" in agent_type:
        raise NotImplementedError("Adversary env not implemented for Iphyre")
    return IphyreNetwork(
        observation_space=env.observation_space,
        action_space=env.action_space,
        obs_type=obs_type,
        should_freeze_embedding=should_freeze_embedding,
        use_ball_relative=use_ball_relative)


def model_for_multigrid_agent(env, agent_type="agent", mlp_hidden=64,
                               recurrent_arch='lstm', recurrent_hidden_size=128):
    if "adversary_env" in agent_type:
        raise NotImplementedError("Adversary env not implemented for MultiGrid")
    return MultigridNetwork(
        observation_space=env.observation_space,
        action_space=env.action_space,
        mlp_hidden=mlp_hidden,
        recurrent_arch=recurrent_arch,
        recurrent_hidden_size=recurrent_hidden_size,
    )


def build_model(env_name, env, agent_type="agent", **kwargs):
    """Return an actor-critic model for the given env."""
    if env_name.startswith("Iphyre"):
        return model_for_iphyre_agent(
            env=env,
            agent_type=agent_type,
            obs_type=kwargs.get("obs_type", "symbolic"),
            should_freeze_embedding=kwargs.get("should_freeze_embedding", False),
            use_ball_relative=kwargs.get("use_ball_relative", False))
    elif env_name.startswith("MultiGrid") or env_name.startswith("MiniGrid"):
        return model_for_multigrid_agent(
            env=env,
            agent_type=agent_type,
            mlp_hidden=kwargs.get("mlp_hidden", 64),
            recurrent_arch=kwargs.get("recurrent_arch", "lstm"),
            recurrent_hidden_size=kwargs.get("recurrent_hidden_size", 128))
    else:
        raise ValueError(f"Unsupported environment {env_name}.")
