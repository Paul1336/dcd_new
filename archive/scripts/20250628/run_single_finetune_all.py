from scripts.exp_utils import (
    run_commands_groups_parallel,
    verify_commands_groups,
)

from train_scripts.make_cmd import generate_cmds

import argparse



def get_commands_groups():

    json_list = [
        ('iphyre/iphyre_single_finetune_dr.json', 'Iphyre-Single-Finetune-DR'),
        ('iphyre/iphyre_single_finetune_plr.json', 'Iphyre-Single-Finetune-PLR'),
        ('iphyre/iphyre_single_finetune_accel.json', 'Iphyre-Single-Finetune-Accel'),
        ('iphyre/iphyre_single_finetune_v_uniform.json', 'Iphyre-Single-Finetune-VUniform'),
        ('iphyre/iphyre_single_finetune_v_accel.json', 'Iphyre-Single-Finetune-VAccel'),
        ('iphyre/iphyre_single_finetune_v_plr.json', 'Iphyre-Single-Finetune-PLR'),
        ('iphyre/iphyre_single_finetune_v_sfl.json', 'Iphyre-Single-Finetune-VSFL'),
        ('iphyre/iphyre_single_finetune_from_scratch.json', 'Iphyre-Single-From-Scratch'),
    ]

    commands_groups = []
    for json_path, method in json_list:
        args = argparse.Namespace(
            dir='train_scripts/grid_configs/',
            json=json_path,
            num_trials=1,
            start_index=0,
            count=True,
            checkpoint=True,
            use_ucb=False,
            xvfb=False,
            method=method,
            exp_name=f'ued_iphyre_finetune_single_full',
            # exp_name="test",
            device="cuda:1",
            log_dir="~/ued-filesystem/logs"
        )

        commands = generate_cmds(args)

        commands_groups.append(commands)
    

    return commands_groups


if __name__ == "__main__":
    commands_groups = get_commands_groups()

    verify_commands_groups(commands_groups)
    run_commands_groups_parallel(commands_groups)
