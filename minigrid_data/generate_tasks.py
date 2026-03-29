"""Generate MinGrid task layouts using Claude API (text-only, no reference images).

Generates wall-obstacle mazes for AdversarialEnv (walls + goal + agent start).
Each task is validated via BFS before saving.

Usage:
    python minigrid_data/generate_tasks.py --num-tasks 100
    python minigrid_data/generate_tasks.py --num-tasks 1000 --size 15 --n-walls 50 --difficulty hard
    python minigrid_data/generate_tasks.py --num-tasks 100 --output-dir minigrid_data/output_test --dry-run

Output per task (in --output-dir/task_NNNNNN/):
    encoding.npy   — (size, size, 3) uint8 numpy array; compatible with AdversarialEnv.reset_to_encoding()
    config.json    — metadata: size, difficulty, path_length, interior_walls, description, ascii
"""

import argparse
import json
import os
import re
import sys
from collections import deque

import numpy as np

# ---------- encoding constants (matches gym_minigrid.minigrid) ----------
#   OBJECT_TO_IDX: unseen=0, empty=1, wall=2, floor=3, door=4, key=5,
#                  ball=6, box=7, goal=8, lava=9, agent=10
#   COLOR_TO_IDX:  red=0, green=1, blue=2, purple=3, yellow=4, grey=5

_EMPTY = np.array([1, 0, 0], dtype=np.uint8)
_WALL  = np.array([2, 5, 0], dtype=np.uint8)   # grey wall
_GOAL  = np.array([8, 1, 0], dtype=np.uint8)   # green goal
_AGENT = np.array([10, 0, 0], dtype=np.uint8)  # agent_id=0, dir=0 (right)

_CHAR_TO_CELL = {'#': _WALL, '.': _EMPTY, 'A': _AGENT, 'G': _GOAL}
_IDX_TO_CHAR  = {1: '.', 2: '#', 8: 'G', 10: 'A'}



def _sanitize_grid_rows(rows, size):
    """Normalize LLM-generated grid rows: fix off-by-one dims and common char substitutions."""
    char_map = {'S': 'A', 's': '.', 'E': 'G', 'e': '.', 'O': '.', 'o': '.', ' ': '.'}

    # Preprocess: skip blanks, truncate/pad each row to exactly `size` chars
    processed = []
    for row in rows:
        if not row.strip():
            continue
        row = row[:size].ljust(size, '#')
        row = ''.join(char_map.get(c, c) for c in row)
        # Force left and right outer walls — valid maze format requires '#' on both ends.
        # This also fixes truncation artifacts where a wider-than-expected row loses its
        # closing '#' after [:size].
        row = '#' + row[1:-1] + '#'
        processed.append(row)

    if len(processed) < size:
        return None
    if len(processed) == size:
        return processed

    # More rows than needed: find a window of `size` rows with top+bottom all-wall borders.
    # The model sometimes emits extra rows above or below the actual grid.
    all_wall = '#' * size
    for start in range(len(processed) - size + 1):
        candidate = processed[start:start + size]
        if candidate[0] == all_wall and candidate[-1] == all_wall:
            return candidate

    # No perfect bordered window found; fall back to last `size` rows
    return processed[-size:]


def _sanitize_grid_rows(rows, size):
    """Normalize LLM-generated grid rows: fix off-by-one dims and common char substitutions."""
    char_map = {'S': 'A', 's': '.', 'E': 'G', 'e': '.', 'O': '.', 'o': '.', ' ': '.'}

    # Preprocess: skip blanks, truncate/pad each row to exactly `size` chars
    processed = []
    for row in rows:
        if not row.strip():
            continue
        row = row[:size].ljust(size, '#')
        row = ''.join(char_map.get(c, c) for c in row)
        # Force left and right outer walls — valid maze format requires '#' on both ends.
        # This also fixes truncation artifacts where a wider-than-expected row loses its
        # closing '#' after [:size].
        row = '#' + row[1:-1] + '#'
        processed.append(row)

    if len(processed) < size:
        return None
    if len(processed) == size:
        return processed

    # More rows than needed: find a window of `size` rows with top+bottom all-wall borders.
    # The model sometimes emits extra rows above or below the actual grid.
    all_wall = '#' * size
    for start in range(len(processed) - size + 1):
        candidate = processed[start:start + size]
        if candidate[0] == all_wall and candidate[-1] == all_wall:
            return candidate

    # No perfect bordered window found; fall back to last `size` rows
    return processed[-size:]

# ---------- ASCII ↔ encoding ----------

def ascii_to_encoding(rows: list, size: int):
    """Convert ASCII grid rows to (size, size, 3) uint8 encoding.

    Returns None if grid is structurally malformed (wrong dimensions or
    unrecognised characters).
    """
    if len(rows) != size:
        return None
    enc = np.zeros((size, size, 3), dtype=np.uint8)
    enc[:, :] = _EMPTY
    for r, row in enumerate(rows):
        row = str(row)
        if len(row) != size:
            return None
        for c, ch in enumerate(row):
            cell = _CHAR_TO_CELL.get(ch)
            if cell is None:
                return None
            enc[c, r] = cell  # col-major: enc[x, y] matches set_encoding's grid.set(i, j) convention
    return enc


def encoding_to_ascii(enc: np.ndarray) -> list:
    """Convert (W, H, 3) col-major encoding back to a list of ASCII row strings."""
    W, H, _ = enc.shape
    rows = []
    for r in range(H):
        rows.append(''.join(_IDX_TO_CHAR.get(int(enc[c, r, 0]), '?') for c in range(W)))
    return rows


# ---------- BFS solvability ----------

def bfs_path_length(enc: np.ndarray) -> int:
    """BFS shortest path from agent to goal (4-directional, walls block).

    Returns path length >= 0, or -1 if no path / agent or goal missing.
    """
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
        (r, c), dist = queue.popleft()
        if (r, c) == goal:
            return dist
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and (nr, nc) not in visited:
                if int(enc[nr, nc, 0]) != 2:   # not wall
                    visited.add((nr, nc))
                    queue.append(((nr, nc), dist + 1))
    return -1


def validate_encoding(enc: np.ndarray, size: int):
    """Structural + BFS validation.

    Returns dict with path_length and interior_walls on success, else None.
    """
    if enc is None:
        return None
    H, W, _ = enc.shape
    if H != size or W != size:
        return None
    # All border cells must be walls
    for i in range(size):
        for pos in [(0, i), (size - 1, i), (i, 0), (i, size - 1)]:
            if int(enc[pos[0], pos[1], 0]) != 2:
                return None
    # Exactly one agent, one goal
    if int(np.sum(enc[:, :, 0] == 10)) != 1:
        return None
    if int(np.sum(enc[:, :, 0] == 8)) != 1:
        return None
    path_len = bfs_path_length(enc)
    if path_len < 0:
        return None
    interior_walls = int(np.sum(enc[1:-1, 1:-1, 0] == 2))
    return {'path_length': path_len, 'interior_walls': interior_walls}


# ---------- OpenAI prompt ----------

def build_prompt(size: int, n_walls: int, difficulty: str) -> str:
    interior = size - 2
    easy_max  = interior
    med_min   = interior
    med_max   = interior * 2
    hard_min  = interior * 2

    diff_instructions = {
        'easy':   (f'EASY difficulty: the shortest path from A to G should be at most {easy_max} steps. '
                   f'Few walls, mostly open space, agent and goal are close.'),
        'medium': (f'MEDIUM difficulty: the shortest path should be between {med_min} and {med_max} steps. '
                   f'Walls form corridors that require the agent to take a non-trivial detour.'),
        'hard':   (f'HARD difficulty: the shortest path should be over {hard_min} steps. '
                   f'Dense walls, long winding corridors, agent and goal placed in opposite corners or far apart.'),
    }
    diff_str = diff_instructions.get(difficulty, diff_instructions['medium'])

    return f"""You are generating maze layouts for a 2D grid navigation task.

GRID RULES:
- Grid size is exactly {size} rows x {size} columns.
- Every border cell (row 0, row {size-1}, col 0, col {size-1}) MUST be '#'.
- Interior cells use exactly these 4 characters:
    '#' = wall (impassable)
    '.' = empty floor (passable)
    'A' = agent start (exactly ONE in the entire grid)
    'G' = goal (exactly ONE in the entire grid)
- Movement is 4-directional (up/down/left/right), no diagonal.
- There MUST be a clear path of passable cells ('.', 'G') from 'A' to 'G'.
- Place approximately {n_walls} interior '#' walls.

DIFFICULTY: {diff_str}

OUTPUT FORMAT — respond with ONLY a JSON object, no markdown, no extra text:
{{
  "grid": [
    "{size * '#'}",
    "# ... row 1 ... #",
    "# ... row 2 ... #",
    ...,
    "{size * '#'}"
  ],
  "description": "one sentence describing this maze",
  "path_length_estimate": <integer>,
  "num_interior_walls": <integer>
}}

The "grid" array must have exactly {size} strings, each exactly {size} characters long."""


# ---------- OpenAI API call ----------

def call_openai(client, size: int, n_walls: int, difficulty: str, model: str, debug: bool = False):
    """Call OpenAI and return parsed JSON dict, or None on failure."""
    prompt = build_prompt(size, n_walls, difficulty)
    response = client.chat.completions.create(
        model=model,
        max_tokens=1024,
        messages=[{'role': 'user', 'content': prompt}],
    )
    text = response.choices[0].message.content.strip()
    if debug:
        print(f'  [debug] raw response ({len(text)} chars):\n{text[:500]}')
    # Strip markdown code fences if the model adds them anyway
    text = re.sub(r'^```[a-z]*\n?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\n?```$', '', text)
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        if debug:
            print(f'  [debug] JSON parse error: {e}')
        return None


def _debug_validate(enc, size: int) -> str:
    """Return a human-readable reason why validation failed."""
    if enc is None:
        return 'ascii_to_encoding returned None (wrong dimensions or unknown chars)'
    H, W, _ = enc.shape
    if H != size or W != size:
        return f'wrong shape {H}x{W}, expected {size}x{size}'
    for i in range(size):
        for pos in [(0, i), (size-1, i), (i, 0), (i, size-1)]:
            if int(enc[pos[0], pos[1], 0]) != 2:
                return f'border cell {pos} is not a wall (type={int(enc[pos[0],pos[1],0])})'
    n_agent = int(np.sum(enc[:, :, 0] == 10))
    n_goal  = int(np.sum(enc[:, :, 0] == 8))
    if n_agent != 1:
        return f'expected 1 agent, found {n_agent}'
    if n_goal != 1:
        return f'expected 1 goal, found {n_goal}'
    if bfs_path_length(enc) < 0:
        return 'BFS: no path from agent to goal'
    return 'ok'


# ---------- task generation + save ----------

def generate_one(client, args, task_id: int, difficulty: str) -> bool:
    """Try up to max_retries times. Returns True if saved successfully."""
    for attempt in range(args.max_retries):
        if args.dry_run:
            # Simulate with a trivial 5-step maze for testing
            s = args.size
            enc = np.zeros((s, s, 3), dtype=np.uint8)
            enc[:] = _EMPTY
            enc[0, :] = enc[s - 1, :] = enc[:, 0] = enc[:, s - 1] = _WALL
            enc[1, 1] = _AGENT
            enc[s - 2, s - 2] = _GOAL
            parsed = {'description': 'dry-run task', 'grid': encoding_to_ascii(enc)}
        else:
            n_walls = {'easy': 10, 'medium': 25, 'hard': 50}.get(difficulty, args.n_walls) if args.difficulty == 'mixed' else args.n_walls
            parsed = call_openai(client, args.size, n_walls, difficulty, args.model,
                                 debug=args.debug)
            if parsed is None:
                if args.debug:
                    print(f'  [debug] attempt {attempt}: call_openai returned None')
                continue
            enc = ascii_to_encoding(_sanitize_grid_rows(parsed.get('grid', []), args.size) or parsed.get('grid', []), args.size)

        metrics = validate_encoding(enc, args.size)
        if metrics is None:
            if args.debug:
                reason = _debug_validate(enc, args.size)
                grid_rows = parsed.get('grid', []) if parsed else []
                print(f'  [debug] attempt {attempt}: validation failed — {reason}')
                if grid_rows:
                    print('  [debug] grid returned by model:')
                    for row in grid_rows[:5]:
                        print(f'    {row}')
                    if len(grid_rows) > 5:
                        print(f'    ... ({len(grid_rows)} rows total)')
            continue

        task_dir = os.path.join(args.output_dir, f'task_{task_id:06d}')
        os.makedirs(task_dir, exist_ok=True)

        np.save(os.path.join(task_dir, 'encoding.npy'), enc)
        config = {
            'size': args.size,
            'difficulty': difficulty,
            'path_length': metrics['path_length'],
            'interior_walls': metrics['interior_walls'],
            'description': parsed.get('description', ''),
            'ascii': parsed.get('grid', encoding_to_ascii(enc)),
        }
        with open(os.path.join(task_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)

        return True

    return False


# ---------- main ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-tasks',   type=int,   default=100,
                        help='Number of tasks to generate')
    parser.add_argument('--size',        type=int,   default=15,
                        help='Grid size (default 15 → 15x15)')
    parser.add_argument('--n-walls',     type=int,   default=50,
                        help='Target number of interior walls')
    parser.add_argument('--difficulty',  default='mixed',
                        choices=['easy', 'medium', 'hard', 'mixed'],
                        help='Difficulty level; mixed cycles easy/medium/hard')
    parser.add_argument('--output-dir',  default='minigrid_data/output')
    parser.add_argument('--model',       default='gpt-4.1-2025-04-14')
    parser.add_argument('--max-retries', type=int,   default=5,
                        help='Retries per task before giving up')
    parser.add_argument('--dry-run',     action='store_true',
                        help='Generate trivial placeholder tasks without calling OpenAI')
    parser.add_argument('--debug',       action='store_true',
                        help='Print raw API responses and validation failure reasons')
    args = parser.parse_args()

    if not args.dry_run:
        try:
            import openai
        except ImportError:
            print('openai package not found. Install: pip install openai')
            sys.exit(1)
        client = openai.OpenAI()
    else:
        client = None
        print('[dry-run] Skipping Claude API calls.')

    difficulties = ['easy', 'medium', 'hard'] if args.difficulty == 'mixed' else [args.difficulty]
    os.makedirs(args.output_dir, exist_ok=True)

    succeeded = 0
    task_id   = _next_task_id(args.output_dir)   # resume-safe: start after existing tasks

    print(f'Generating {args.num_tasks} tasks (size={args.size}, n_walls={args.n_walls}, '
          f'difficulty={args.difficulty}) → {args.output_dir}')
    print(f'Starting at task_id={task_id}')

    for i in range(args.num_tasks):
        diff = difficulties[i % len(difficulties)]
        ok = generate_one(client, args, task_id, diff)
        if ok:
            succeeded += 1
            task_id += 1
            if succeeded % 10 == 0 or succeeded == args.num_tasks:
                print(f'  {succeeded}/{args.num_tasks} saved')
        else:
            print(f'  [warn] task {i}: failed after {args.max_retries} retries, skipping')

    print(f'\nDone. {succeeded}/{args.num_tasks} tasks saved to {args.output_dir}')


def _next_task_id(output_dir: str) -> int:
    """Return the next available task_id by scanning existing task_NNNNNN dirs."""
    if not os.path.isdir(output_dir):
        return 0
    ids = []
    for name in os.listdir(output_dir):
        if name.startswith('task_') and os.path.isdir(os.path.join(output_dir, name)):
            try:
                ids.append(int(name[5:]))
            except ValueError:
                pass
    return (max(ids) + 1) if ids else 0


if __name__ == '__main__':
    main()
