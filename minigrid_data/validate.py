"""Validate and report statistics for generated MinGrid tasks.

Usage:
    python minigrid_data/validate.py --task-dir minigrid_data/output
    python minigrid_data/validate.py --task-dir minigrid_data/output --filter-min-path 5 --delete-invalid
    python minigrid_data/validate.py --task-dir minigrid_data/output --print-ascii task_000003
"""

import argparse
import json
import os
import shutil
from collections import Counter, deque

import numpy as np

_IDX_TO_CHAR = {1: '.', 2: '#', 8: 'G', 10: 'A'}


def bfs_path_length(enc: np.ndarray) -> int:
    """BFS from agent (type=10) to goal (type=8). Returns length or -1."""
    H, W, _ = enc.shape
    start = goal = None
    for r in range(H):
        for c in range(W):
            t = int(enc[r, c, 0])
            if t == 10:
                start = (r, c)
            elif t == 8:
                goal = (r, c)
    if start is None or goal is None:
        return -1
    visited = {start}
    queue = deque([(start, 0)])
    while queue:
        (r, c), d = queue.popleft()
        if (r, c) == goal:
            return d
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and (nr, nc) not in visited:
                if int(enc[nr, nc, 0]) != 2:
                    visited.add((nr, nc))
                    queue.append(((nr, nc), d + 1))
    return -1


def print_ascii(enc: np.ndarray) -> None:
    H, W, _ = enc.shape
    for r in range(H):
        print(''.join(_IDX_TO_CHAR.get(int(enc[r, c, 0]), '?') for c in range(W)))


def load_tasks(task_dir: str) -> list:
    tasks = []
    for name in sorted(os.listdir(task_dir)):
        d = os.path.join(task_dir, name)
        enc_path = os.path.join(d, 'encoding.npy')
        cfg_path = os.path.join(d, 'config.json')
        if not (os.path.isfile(enc_path) and os.path.isfile(cfg_path)):
            continue
        enc = np.load(enc_path)
        with open(cfg_path) as f:
            cfg = json.load(f)
        tasks.append((name, enc, cfg))
    return tasks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task-dir',        required=True,
                        help='Directory containing task_NNNNNN subdirectories')
    parser.add_argument('--filter-min-path', type=int, default=0,
                        help='Exclude tasks with BFS path length below this')
    parser.add_argument('--filter-max-path', type=int, default=9999,
                        help='Exclude tasks with BFS path length above this')
    parser.add_argument('--recompute-bfs',   action='store_true',
                        help='Recompute BFS from encoding (ignore saved path_length)')
    parser.add_argument('--delete-invalid',  action='store_true',
                        help='Delete task directories that fail BFS or path-length filter')
    parser.add_argument('--print-ascii',     default=None, metavar='TASK_NAME',
                        help='Print ASCII grid for a specific task name (e.g. task_000003)')
    args = parser.parse_args()

    # Single task inspection mode
    if args.print_ascii:
        d = os.path.join(args.task_dir, args.print_ascii)
        enc_path = os.path.join(d, 'encoding.npy')
        cfg_path = os.path.join(d, 'config.json')
        if not os.path.isfile(enc_path):
            print(f'encoding.npy not found in {d}')
            return
        enc = np.load(enc_path)
        print_ascii(enc)
        if os.path.isfile(cfg_path):
            with open(cfg_path) as f:
                cfg = json.load(f)
            print(f'\ndifficulty={cfg.get("difficulty")}  '
                  f'path_length={cfg.get("path_length")}  '
                  f'interior_walls={cfg.get("interior_walls")}')
            print(f'description: {cfg.get("description", "")}')
        print(f'BFS (recomputed): {bfs_path_length(enc)}')
        return

    tasks = load_tasks(args.task_dir)
    print(f'Loaded {len(tasks)} tasks from {args.task_dir}')

    path_lengths = []
    wall_counts  = []
    diff_counter = Counter()
    invalid_names = []

    for name, enc, cfg in tasks:
        path_len = (bfs_path_length(enc)
                    if args.recompute_bfs
                    else cfg.get('path_length', bfs_path_length(enc)))

        if path_len < 0:
            print(f'  [unsolvable] {name}')
            invalid_names.append(name)
            continue

        if not (args.filter_min_path <= path_len <= args.filter_max_path):
            print(f'  [filtered]   {name}  path={path_len}')
            invalid_names.append(name)
            continue

        path_lengths.append(path_len)
        wall_counts.append(cfg.get('interior_walls', 0))
        diff_counter[cfg.get('difficulty', '?')] += 1

    valid = len(path_lengths)
    total = len(tasks)
    print(f'\nValid: {valid} / {total}  ({100*valid//total if total else 0}%)')

    if path_lengths:
        print(f'Path length    — min={min(path_lengths):4d}  max={max(path_lengths):4d}'
              f'  mean={sum(path_lengths)/valid:.1f}')
        print(f'Interior walls — min={min(wall_counts):4d}  max={max(wall_counts):4d}'
              f'  mean={sum(wall_counts)/valid:.1f}')
        print(f'Difficulty: {dict(diff_counter)}')

        # Histogram of path lengths
        if valid >= 5:
            buckets = [0] * 10
            lo, hi = min(path_lengths), max(path_lengths)
            span = max(hi - lo, 1)
            for p in path_lengths:
                idx = min(int((p - lo) / span * 10), 9)
                buckets[idx] += 1
            print('\nPath-length histogram:')
            bucket_size = span / 10
            for i, cnt in enumerate(buckets):
                label = f'{lo + i*bucket_size:.0f}-{lo + (i+1)*bucket_size:.0f}'
                bar = '#' * (cnt * 40 // valid)
                print(f'  {label:>8}  {bar}  ({cnt})')

    if args.delete_invalid and invalid_names:
        for name in invalid_names:
            d = os.path.join(args.task_dir, name)
            shutil.rmtree(d)
        print(f'\nDeleted {len(invalid_names)} invalid/filtered task directories.')


if __name__ == '__main__':
    main()
