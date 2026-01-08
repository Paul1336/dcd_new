from scripts.exp_utils import (
    run_commands_groups_parallel,
    verify_commands_groups,
)

from train_scripts.make_cmd import generate_cmds

import argparse



def get_commands_groups():

    json_list = [
        ('iphyre/iphyre_dr.json', 'Iphyre-DR'),
        ('iphyre/iphyre_robust_plr.json', 'Iphyre-Robust-PLR'),
        ('iphyre/iphyre_accel.json', 'Iphyre-Accel'),
        ('iphyre/iphyre_v_uniform.json', 'Iphyre-V-Uniform'),
        ('iphyre/iphyre_v_robust_plr.json', 'Iphyre-V-Robust-PLR'),
        ('iphyre/iphyre_v_accel.json', 'Iphyre-V-Accel'),
    ]

    commands_groups = []
    for json_path, method in json_list:
        args = argparse.Namespace(
            dir='train_scripts/grid_configs/',
            json=json_path,
            num_trials=3,
            start_index=0,
            count=True,
            checkpoint=False,
            use_ucb=False,
            xvfb=False,
            method=method,
            exp_name=f'ued_iphyre_64_full',
            device="cuda:1"
        )

        commands = generate_cmds(args)
        
        commands_groups.append(commands)

    return commands_groups


if __name__ == "__main__":
    commands_groups = get_commands_groups()

    verify_commands_groups(commands_groups)
    run_commands_groups_parallel(commands_groups)
