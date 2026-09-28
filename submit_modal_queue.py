"""Submit persistent, independent Modal calls for missing GroupARC conditions."""
import argparse
import json
from pathlib import Path

import modal
import wandb

from modal_runner import APP_NAME, SEED, select_successful_baselines


VARIANTS = ('original', 'projected_conv', 'pure_conv')
PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wandb-entity', default='arc_agi')
    parser.add_argument('--wandb-project', default='grouparc')
    parser.add_argument('--baseline-project', default='compressarc')
    parser.add_argument('--baseline-tag', default='full-ablation')
    parser.add_argument('--split', default='training')
    parser.add_argument('--iterations', type=int, default=1500)
    parser.add_argument('--dry-run', action='store_true')
    return parser.parse_args()


def finished_conditions(runs, split, iterations):
    """Return successful task/variant pairs safe to omit from resubmission."""
    finished = set()
    for run in runs:
        config = run.config
        if (run.state == 'finished'
                and config.get('split') == split
                and config.get('iterations') == iterations
                and config.get('multitensor_constraints') == 'strict'
                and config.get('shift_variant') in VARIANTS):
            finished.add((config.get('task_name'), config['shift_variant']))
    return finished


def main():
    args = parse_args()
    api = wandb.Api()
    source_project = f'{args.wandb_entity}/{args.baseline_project}'
    destination_project = f'{args.wandb_entity}/{args.wandb_project}'
    baseline_runs = api.runs(source_project, filters={'$and': [
        {'tags': 'modal'}, {'tags': 'strict-vs-relaxed'}, {'tags': args.baseline_tag},
    ]})
    selected = select_successful_baselines(baseline_runs, args.split)
    if not selected:
        raise RuntimeError(f'No successful baseline runs found in {source_project}')
    prior_runs = api.runs(destination_project, filters={'tags': 'convolution-ablation'})
    finished = finished_conditions(prior_runs, args.split, args.iterations)
    conditions = [dict(
        task_id=task,
        constraint_policy='strict',
        shift_variant=variant,
        split=args.split,
        seed=SEED,
        number_of_tasks=len(selected),
        iterations=args.iterations,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        execution_tag='convolution-ablation',
        experiment='shift-convolution',
        source_run_ids=source_ids,
    ) for task, source_ids in selected.items() for variant in VARIANTS
      if (task, variant) not in finished]
    print(f'Skipping {len(finished)} finished conditions.')
    print(f'Submitting {len(conditions)} independent calls to one T4 worker.')

    calls = []
    if not args.dry_run:
        function = modal.Function.from_name(APP_NAME, 'run_condition')
        for condition in conditions:
            call = function.spawn(condition)
            calls.append({
                'function_call_id': call.object_id,
                'dashboard_url': call.get_dashboard_url(),
                'task_id': condition['task_id'],
                'shift_variant': condition['shift_variant'],
            })
            print(f"Submitted {condition['task_id']} / {condition['shift_variant']}: {call.object_id}")

    manifest = {
        'modal_app': APP_NAME,
        'source_project': source_project,
        'destination_project': destination_project,
        'skipped_finished': sorted([list(item) for item in finished]),
        'conditions': conditions,
        'calls': calls,
    }
    path = PROJECT_ROOT / 'deployed_queue_manifest.json'
    path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(f'Manifest: {path}')


if __name__ == '__main__':
    main()
