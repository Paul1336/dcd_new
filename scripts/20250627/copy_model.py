import os
import sys

DR = [
    "ued--domain_randomization-lr0.00025-epoch30-mb4-v0.5-gc0.5-henv0.01-ha0.01-tl_2_20250622-052619_312",
    "ued--domain_randomization-lr0.00025-epoch30-mb4-v0.5-gc0.5-henv0.01-ha0.01-tl_1_20250622-052619_684",
    "ued--domain_randomization-lr0.00025-epoch30-mb4-v0.5-gc0.5-henv0.01-ha0.01-tl_0_20250622-052619_13",
]
PLR = [
    "ued--domain_randomization-noexpgrad-lr0.00025-epoch30-mb4-v0.5-gc0.5-henv0.01-ha0.01-plr0.5-rho0.5-n1000-st0.5-positive_value_loss-rank-t0.1-tl_1_20250622-052706_622",
    "ued--domain_randomization-noexpgrad-lr0.00025-epoch30-mb4-v0.5-gc0.5-henv0.01-ha0.01-plr0.5-rho0.5-n1000-st0.5-positive_value_loss-rank-t0.1-tl_2_20250622-052706_985",
    "ued--domain_randomization-noexpgrad-lr0.00025-epoch30-mb4-v0.5-gc0.5-henv0.01-ha0.01-plr0.5-rho0.5-n1000-st0.5-positive_value_loss-rank-t0.1-tl_0_20250622-052706_757",
]
ACCEL = [
    "ued--domain_randomization-noexpgrad-lr0.00025-epoch30-mb4-v0.5-gc0.5-henv0.01-ha0.01-plr0.9-rho0.5-n1000-st0.5-positive_value_loss-rank-t0.1-editor1.0-random-n1-baseeasy-tl_0_20250622-062233_502",
    "ued--domain_randomization-noexpgrad-lr0.00025-epoch30-mb4-v0.5-gc0.5-henv0.01-ha0.01-plr0.9-rho0.5-n1000-st0.5-positive_value_loss-rank-t0.1-editor1.0-random-n1-baseeasy-tl_2_20250622-062233_806",
    "ued--domain_randomization-noexpgrad-lr0.00025-epoch30-mb4-v0.5-gc0.5-henv0.01-ha0.01-plr0.9-rho0.5-n1000-st0.5-positive_value_loss-rank-t0.1-editor1.0-random-n1-baseeasy-tl_1_20250622-062233_318",
]
VUniform = [
    "Iphyre-V-Uniform-10k_20250623-025232_797",
    "Iphyre-V-Uniform-10k_20250623-025232_58",
    "Iphyre-V-Uniform-10k_20250623-025232_812",
]
VAccel = [
    "Iphyre-V-Accel-10k_20250623-025301_820",
    "Iphyre-V-Accel-10k_20250623-025301_635",
    "Iphyre-V-Accel-10k_20250623-025301_350",
]
VPLR = [
    "Iphyre-V-Robust-PLR-10k_20250623-025647_903",
    "Iphyre-V-Robust-PLR-10k_20250623-025646_641",
    "Iphyre-V-Robust-PLR-10k_20250623-025646_682",
]
VSFL = [
    "Iphyre-V-SFL-0.3-10k_20250625-082050_998",
    "Iphyre-V-SFL-0.3-10k_20250625-082050_187",
    "Iphyre-V-SFL-0.3-10k_20250625-082050_56",
]
SFL = [
    "Iphyre-SFL_20250628-031918_668",
    "Iphyre-SFL_20250628-031918_770",
    "Iphyre-SFL_20250628-031918_800",
]

run_sets = [
    ("DR", DR),
    ("PLR", PLR),
    ("ACCEL", ACCEL),
    ("VUniform", VUniform),
    ("VAccel", VAccel),
    ("VPLR", VPLR),
    ("VSFL", VSFL),
    ("SFL", SFL),
]


for run_set_name, run_set in run_sets:
    for index, run_name in enumerate(run_set):
        checkpoint_name = "checkpoint_29491200.pt"
        os.system(f"cp -r ~/projects/ued-data/{run_name}/checkpoints/{checkpoint_name} ./scripts/20250627/{run_set_name}_{index}.pt")

