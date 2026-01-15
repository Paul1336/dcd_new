from .sfl.SFLRunner import SFLRunner

def create_runner(
    args,
    venv,
    agent, 
    ued_venv, 
    adversary_agent,
    adversary_env,
    train=True,
    plr_args=None,
    flexible_protagonist=False,):
    if args.ued_algo == 'sfl':
        return SFLRunner(
            args,
            venv,
            agent,
            ued_venv,
            adversary_agent,
            adversary_env,
            train,
            plr_args,
            flexible_protagonist)
    else:
        raise ValueError(f"UED algorithm {args.ued_algo} not implemented.")
