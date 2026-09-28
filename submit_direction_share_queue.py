"""Submit persistent Modal calls for the three-way D4 direction-share ablation."""
import argparse
import json
from pathlib import Path

import modal
import wandb

from modal_runner import APP_NAME, SEED, select_successful_baselines


VARIANTS = ('projected_d4', 'pure_d4')
PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wandb-entity', default='arc_agi')
    parser.add_argument('--wandb-project', default='grouparc')
    parser.add_argument('--baseline-project', default='compressarc')
    parser.add_argument('--baseline-tag', default='full-ablation')
    parser.add_argument('--split', default='training')
    parser.add_argument('--iterations', type=int, default=1500)
    parser.add_argument('--early-stop-window', type=int, default=50)
    parser.add_argument('--early-stop-epsilon', type=float, default=1e-4)
    parser.add_argument('--early-stop-warmup', type=int, default=300)
    parser.add_argument('--dry-run', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    api = wandb.Api()
    source_project = f'{args.wandb_entity}/{args.baseline_project}'
    destination_project = f'{args.wandb_entity}/{args.wandb_project}'
    baselines = api.runs(source_project, filters={'$and': [
        {'tags': 'modal'}, {'tags': 'strict-vs-relaxed'}, {'tags': args.baseline_tag},
    ]})
    selected = select_successful_baselines(baselines, args.split)
    prior = api.runs(destination_project, filters={'tags': 'direction-share-ablation'})
    finished = {
        (run.config.get('task_name'), run.config.get('direction_share_variant'))
        for run in prior
        if run.state == 'finished'
        and run.config.get('split') == args.split
        and run.config.get('iterations') == args.iterations
        and run.config.get('shift_variant') == 'original'
    }
    conditions = [dict(
        task_id=task, constraint_policy='strict', shift_variant='original',
        direction_share_variant=variant, split=args.split, seed=SEED,
        number_of_tasks=len(selected), iterations=args.iterations,
        wandb_project=args.wandb_project, wandb_entity=args.wandb_entity,
        execution_tag='direction-share-ablation', experiment='d4-direction-share',
        source_run_ids=source_ids, early_stop=True,
        early_stop_window=args.early_stop_window,
        early_stop_epsilon=args.early_stop_epsilon,
        early_stop_warmup=args.early_stop_warmup,
    ) for task, source_ids in selected.items() for variant in VARIANTS
      if (task, variant) not in finished]
    print(f'Skipping {len(finished)} finished conditions; submitting {len(conditions)}.')
    calls = []
    if not args.dry_run:
        function = modal.Function.from_name(APP_NAME, 'run_condition')
        for condition in conditions:
            call = function.spawn(condition)
            calls.append({
                'function_call_id': call.object_id,
                'dashboard_url': call.get_dashboard_url(),
                'task_id': condition['task_id'],
                'direction_share_variant': condition['direction_share_variant'],
            })
            print(f"Submitted {condition['task_id']} / {condition['direction_share_variant']}")
    manifest = {
        'modal_app': APP_NAME,
        'source_project': source_project,
        'destination_project': destination_project,
        'early_stop': {
            'window': args.early_stop_window,
            'relative_epsilon': args.early_stop_epsilon,
            'warmup': args.early_stop_warmup,
        },
        'skipped_finished': sorted([list(item) for item in finished]),
        'conditions': conditions,
        'calls': calls,
    }
    path = PROJECT_ROOT / 'direction_share_queue_manifest.json'
    path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(f'Manifest: {path}')


if __name__ == '__main__':
    main()
