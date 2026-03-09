"""Set the group of wandb runs that have a given tag.

Usage:
    python scripts/wandb_set_group_by_tag.py --tag foo --group foo
    python scripts/wandb_set_group_by_tag.py --tag foo --group bar --entity myteam --project myproject
    python scripts/wandb_set_group_by_tag.py --tag foo --group foo --dry-run
"""

import argparse
import wandb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag',     required=True, help='Filter runs that contain this tag')
    parser.add_argument('--group',   required=True, help='Group name to assign')
    parser.add_argument('--entity',  default=None,  help='W&B entity (defaults to logged-in user)')
    parser.add_argument('--project', default=None,  help='W&B project (required if not set as default)')
    parser.add_argument('--dry-run', action='store_true', help='Print matching runs without updating')
    args = parser.parse_args()

    api = wandb.Api()

    path = '/'.join(filter(None, [args.entity, args.project]))
    if not path:
        raise ValueError('Provide at least --project (and optionally --entity)')

    runs = api.runs(path, filters={'tags': {'$in': [args.tag]}})

    matched = list(runs)
    if not matched:
        print(f'No runs found with tag "{args.tag}" in {path}')
        return

    print(f'Found {len(matched)} run(s) with tag "{args.tag}":')
    for run in matched:
        print(f'  {run.id}  name={run.name!r}  current_group={run.group!r}')

    if args.dry_run:
        print('\n[dry-run] No changes made.')
        return

    print(f'\nSetting group → "{args.group}" ...')
    for run in matched:
        run.group = args.group
        run.update()
        print(f'  updated {run.id}')

    print('Done.')


if __name__ == '__main__':
    main()
