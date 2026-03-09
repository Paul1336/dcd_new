"""Batch-generate MinGrid tasks in parallel worker processes.

Splits --num-tasks across --workers processes.  Each worker runs
generate_tasks.py with a separate output subdirectory; results can be
merged with merge_outputs.py or referenced directly via a dir-list.

Usage:
    python minigrid_data/batch_generate.py --num-tasks 4000 --workers 8
    python minigrid_data/batch_generate.py --num-tasks 4000 --workers 8 --difficulty hard --size 15
"""

import argparse
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed


def _run_worker(worker_id: int, num_tasks: int, args_extra: list, base_dir: str) -> int:
    out_dir = os.path.join(base_dir, f'worker_{worker_id:03d}')
    os.makedirs(out_dir, exist_ok=True)
    cmd = [
        sys.executable,
        os.path.join(os.path.dirname(__file__), 'generate_tasks.py'),
        '--num-tasks', str(num_tasks),
        '--output-dir', out_dir,
    ] + args_extra
    result = subprocess.run(cmd, check=False)
    return result.returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-tasks',   type=int,  default=1000)
    parser.add_argument('--workers',     type=int,  default=4)
    parser.add_argument('--output-dir',  default='minigrid_data/output_batch')
    parser.add_argument('--size',        type=int,  default=15)
    parser.add_argument('--n-walls',     type=int,  default=50)
    parser.add_argument('--difficulty',  default='mixed',
                        choices=['easy', 'medium', 'hard', 'mixed'])
    parser.add_argument('--model',       default='claude-sonnet-4-6')
    parser.add_argument('--max-retries', type=int,  default=5)
    args = parser.parse_args()

    tasks_per_worker = args.num_tasks // args.workers
    remainder = args.num_tasks % args.workers

    extra = [
        '--size',        str(args.size),
        '--n-walls',     str(args.n_walls),
        '--difficulty',  args.difficulty,
        '--model',       args.model,
        '--max-retries', str(args.max_retries),
    ]

    os.makedirs(args.output_dir, exist_ok=True)
    print(f'Launching {args.workers} workers, '
          f'~{tasks_per_worker} tasks each → {args.output_dir}')

    futures = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for wid in range(args.workers):
            n = tasks_per_worker + (1 if wid < remainder else 0)
            f = pool.submit(_run_worker, wid, n, extra, args.output_dir)
            futures[f] = wid

        for f in as_completed(futures):
            wid = futures[f]
            rc = f.result()
            status = 'ok' if rc == 0 else f'exit={rc}'
            print(f'  worker {wid:03d} done ({status})')

    # Print summary: count tasks across all worker subdirs
    total = 0
    for wid in range(args.workers):
        d = os.path.join(args.output_dir, f'worker_{wid:03d}')
        if os.path.isdir(d):
            total += sum(1 for n in os.listdir(d)
                         if n.startswith('task_') and
                         os.path.isfile(os.path.join(d, n, 'encoding.npy')))

    print(f'\nTotal tasks saved: {total}')
    print(f'Output dirs: {args.output_dir}/worker_000 ... worker_{args.workers-1:03d}')


if __name__ == '__main__':
    main()
