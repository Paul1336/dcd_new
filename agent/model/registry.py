from .iphyre import IphyreNetwork


def model_for_iphyre_agent(env, agent_type="agent", obs_type="symbolic", should_freeze_embedding=False):
    if "adversary_env" in agent_type:
        raise NotImplementedError("Adversary env not implemented for Iphyre")
    return IphyreNetwork(
        observation_space=env.observation_space,
        action_space=env.action_space,
        obs_type=obs_type,
        should_freeze_embedding=should_freeze_embedding)


def build_model(env_name, env, agent_type="agent", **kwargs):
    """Return an actor-critic model for the given env."""
    if env_name.startswith("Iphyre"):
        return model_for_iphyre_agent(
            env=env,
            agent_type=agent_type,
            obs_type=kwargs.get("obs_type", "symbolic"),
            should_freeze_embedding=kwargs.get("should_freeze_embedding", False))
    else:
        raise ValueError(f"Unsupported environment {env_name}.")
