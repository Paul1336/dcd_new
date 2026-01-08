from scripts.exp_utils import (
    run_commands_groups_parallel,
    verify_commands_groups,
)

from train_scripts.make_cmd import generate_cmds

import argparse



def get_commands_groups():

    json_list = [
        ('minigrid/60_blocks_uniform/mg_60b_uni_dr.json', 'MG-60B-DR'),
        ('minigrid/60_blocks_uniform/mg_60b_uni_accel_empty.json', 'MG-60B-ACCEL'),
        ('minigrid/60_blocks_uniform/mg_60b_uni_paired.json', 'MG-60B-PAIRED'),
        ('minigrid/60_blocks_uniform/mg_60b_uni_robust_plr.json', 'MG-60B-PLR'),
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
            exp_name=f'ued_minigrid'
        )

        commands = generate_cmds(args)
        
        commands_groups.append(commands)

    return commands_groups


if __name__ == "__main__":
    commands_groups = get_commands_groups()

    verify_commands_groups(commands_groups)
    run_commands_groups_parallel(commands_groups)
