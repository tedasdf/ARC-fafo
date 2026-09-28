"""Submit projected and pure LSE cummax replacements to the deployed Modal app."""
import argparse
import json
from pathlib import Path

import modal
import wandb

from modal_runner import APP_NAME, SEED, select_successful_baselines


VARIANTS = ('projected_lse', 'pure_lse')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wandb-entity', default='arc_agi')
    parser.add_argument('--wandb-project', default='grouparc')
    parser.add_argument('--baseline-project', default='compressarc')
    parser.add_argument('--iterations', type=int, default=1500)
    parser.add_argument('--early-stop-window', type=int, default=50)
    parser.add_argument('--early-stop-epsilon', type=float, default=1e-4)
    parser.add_argument('--early-stop-warmup', type=int, default=300)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    api = wandb.Api()
    source = f'{args.wandb_entity}/{args.baseline_project}'
    destination = f'{args.wandb_entity}/{args.wandb_project}'
    baselines = api.runs(source, filters={'$and': [
        {'tags': 'modal'}, {'tags': 'strict-vs-relaxed'}, {'tags': 'full-ablation'},
    ]})
    selected = select_successful_baselines(baselines, 'training')
    prior = api.runs(destination, filters={'tags': 'lse-ablation'})
    finished = {(run.config.get('task_name'), run.config.get('cummax_variant'))
                for run in prior if run.state == 'finished'
                and run.config.get('iterations') == args.iterations}
    conditions = [dict(
        task_id=task, constraint_policy='strict', shift_variant='original',
        direction_share_variant='original', cummax_variant=variant,
        split='training', seed=SEED, number_of_tasks=len(selected),
        iterations=args.iterations, wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity, execution_tag='lse-ablation',
        experiment='lse-cummax', source_run_ids=source_ids, early_stop=True,
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
                'cummax_variant': condition['cummax_variant'],
            })
            print(f"Submitted {condition['task_id']} / {condition['cummax_variant']}")
    manifest = {
        'modal_app': APP_NAME, 'source_project': source,
        'destination_project': destination, 'conditions': conditions, 'calls': calls,
        'shared_baseline': 'shift_original / direction_share_original / cummax_original',
    }
    path = Path(__file__).resolve().parent / 'lse_queue_manifest.json'
    path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(f'Manifest: {path}')


if __name__ == '__main__':
    main()
