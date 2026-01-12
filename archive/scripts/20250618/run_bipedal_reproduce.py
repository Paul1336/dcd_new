from scripts.exp_utils import (
    run_commands_groups_parallel,
    verify_commands_groups,
)

from train_scripts.make_cmd import generate_cmds

import argparse



def get_commands_groups():

    json_list = [
        ('bipedal/bipedal_accel.json', 'Bipedal-Accel'),
        ('bipedal/bipedal_dr.json', 'Bipedal-DR'),
        ('bipedal/bipedal_minimax.json', 'Bipedal-Minimax'),
        ('bipedal/bipedal_paired.json', 'Bipedal-Paired'),
        ('bipedal/bipedal_robust_plr.json', 'Bipedal-Robust-PLR'),
        ('bipedal/bipedal_alpgmm.json', 'Bipedal-ALPGMM'),
        ('bipedal/bipedal_accel_poet.json', 'Bipedal-Accel-Poet'),
    ]

    commands_groups = []
    for json_path, method in json_list:
        args = argparse.Namespace(
            dir='train_scripts/grid_configs/',
            json=json_path,
            num_trials=5,
            start_index=0,
            count=True,
            checkpoint=False,
            use_ucb=False,
            xvfb=False,
            method=method,
            exp_name=f'bipedal_reproduce_{method}'
        )

        commands = generate_cmds(args)
        
        commands_groups.append(commands)

    return commands_groups


if __name__ == "__main__":
    commands_groups = get_commands_groups()

    verify_commands_groups(commands_groups)
    run_commands_groups_parallel(commands_groups)
