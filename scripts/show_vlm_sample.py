"""Show which VLM task directories are sampled into MultiGrid-VLMSampled-v0.

Replicates the exact sampling logic in env/benchmark/minigrid/suites.py
without importing any training code.

Usage:
    python scripts/show_vlm_sample.py
    python scripts/show_vlm_sample.py --num-tasks 20 --base-seed 0
"""

import argparse
import os
import random

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-tasks', type=int, default=20)
    parser.add_argument('--base-seed', type=int, default=0)
    parser.add_argument('--task-dir', type=str, default='minigrid_data/output')
    args = parser.parse_args()

    task_dir = args.task_dir
    if not os.path.isdir(task_dir):
        print(f'Directory not found: {task_dir}')
        return

    # Mirror load_vlm_gen_tasks but track paths
    entries = []
    for name in sorted(os.listdir(task_dir)):
        d = os.path.join(task_dir, name)
        enc_path = os.path.join(d, 'encoding.npy')
        if os.path.isfile(enc_path):
            entries.append((enc_path, np.load(enc_path)))

    print(f'Total encodings in pool: {len(entries)}')

    rng = random.Random(args.base_seed)
    if len(entries) > args.num_tasks:
        sampled = rng.sample(entries, args.num_tasks)
    else:
        sampled = list(entries)

    print(f'\nSampled {len(sampled)} levels (base_seed={args.base_seed}):\n')
    for i, (path, enc) in enumerate(sampled):
        print(f'  vlm_{i:04d}  {path}  shape={enc.shape}')


if __name__ == '__main__':
    main()
