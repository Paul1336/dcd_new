from .sfl.sfl_runner import SFLRunner
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
):
    if args.ued_algo in ('sfl', 'old_sfl', 'domain_randomization'):
        agents = {AgentRole.AGENT: agent}
        return SFLRunner(
            args=args,
            venv=venv,
            agents=agents,
            ued_venv=ued_venv,
            train=train,
        )
    else:
        raise ValueError(f"UED algorithm '{args.ued_algo}' not implemented.")