from .sfl.sfl_runner import SFLRunner
from .accel.accel_runner import ACCELRunner
from .plr.plr_runner import PLRRunner
from .runner import AgentRole


def create_runner(
    args,
    venv,
    agent,
    ued_venv=None,
    adversary_agent=None,
    adversary_env=None,
    train=True,
    plr_args=None,
    flexible_protagonist=False,
    obs_encoder=None,
):
    if args.ued_algo in ('sfl', 'old_sfl', 'domain_randomization'):
        agents = {AgentRole.AGENT: agent}
        return SFLRunner(
            args=args,
            venv=venv,
            agents=agents,
            ued_venv=ued_venv,
            train=train,
            obs_encoder=obs_encoder,
        )
    elif args.ued_algo == 'plr':
        agents = {AgentRole.AGENT: agent}
        return PLRRunner(
            args=args,
            venv=venv,
            agents=agents,
            ued_venv=ued_venv,
            train=train,
        )
    elif args.ued_algo == 'accel':
        agents = {AgentRole.AGENT: agent}
        if getattr(args, 'use_accel_paired', False) and adversary_agent is not None:
            agents[AgentRole.ADVERSARY_AGENT] = adversary_agent
        return ACCELRunner(
            args=args,
            venv=venv,
            agents=agents,
            ued_venv=ued_venv,
            train=train,
        )
    else:
        raise ValueError(f"UED algorithm '{args.ued_algo}' not implemented.")