"""Run a seeded strict-vs-relaxed CompressARC ablation on Modal."""

import json
import random
import shlex
import subprocess
import sys
from pathlib import Path

import modal


SEED = 42
SPLIT = 'training'
NUMBER_OF_TASKS = 20
CONSTRAINT_POLICIES = ('strict', 'relaxed')
REMOTE_PROJECT_DIR = '/root/CompressARC'

project_root = Path(__file__).resolve().parent

image = (
    modal.Image.debian_slim(python_version='3.11')
    .pip_install_from_requirements(str(project_root / 'requirements.txt'))
    .env({
        'MPLBACKEND': 'Agg',
        'PYTHONUNBUFFERED': '1',
    })
    .add_local_dir(
        project_root,
        remote_path=REMOTE_PROJECT_DIR,
        ignore=[
            '.git/**',
            '.venv/**',
            '.runtime/**',
            '__pycache__/**',
            'wandb/**',
            'notebook/**',
            'results_for_the_blog_post/**',
            '*.pyc',
        ],
    )
)

APP_NAME = 'compressarc-strict-relaxed-ablation'
app = modal.App(APP_NAME)


@app.function(
    image=image,
    gpu='T4',
    secrets=[modal.Secret.from_name('wandb-secret', required_keys=['WANDB_API_KEY'])],
    timeout=2 * 60 * 60,
    max_containers=1,
)
def run_condition(condition):
    """Run one task and one constraint policy in an isolated GPU container."""
    task_id = condition['task_id']
    constraint_policy = condition['constraint_policy']
    split = condition['split']
    seed = condition['seed']
    number_of_tasks = condition['number_of_tasks']
    iterations = condition['iterations']
    wandb_project = condition['wandb_project']
    execution_tag = condition['execution_tag']
    shift_variant = condition.get('shift_variant', 'original')
    direction_share_variant = condition.get('direction_share_variant', 'original')
    cummax_variant = condition.get('cummax_variant', 'original')

    command = [
        sys.executable,
        'analyze_example_wandb.py',
        '--split', split,
        '--task', task_id,
        '--iterations', str(iterations),
        '--wandb',
        '--wandb-project', wandb_project,
        '--wandb-tags',
        f'random-{number_of_tasks}',
        f'sample-seed-{seed}',
        condition.get('experiment', 'strict-vs-relaxed'),
        'modal',
        execution_tag,
        '--multitensor-constraints', constraint_policy,
        '--shift-variant', shift_variant,
        '--direction-share-variant', direction_share_variant,
        '--cummax-variant', cummax_variant,
    ]
    if condition.get('early_stop'):
        command.extend([
            '--early-stop',
            '--early-stop-window', str(condition.get('early_stop_window', 50)),
            '--early-stop-epsilon', str(condition.get('early_stop_epsilon', 1e-4)),
            '--early-stop-warmup', str(condition.get('early_stop_warmup', 300)),
        ])
    if condition.get('wandb_entity'):
        command.extend(['--wandb-entity', condition['wandb_entity']])
    if condition.get('source_run_ids'):
        # Tags keep each rerun traceable to the baseline selection evidence.
        tag_index = command.index('--multitensor-constraints')
        command[tag_index:tag_index] = [f'baseline-{rid}' for rid in condition['source_run_ids']]

    print(f'Starting {task_id} ({constraint_policy})')
    print(shlex.join(command))
    subprocess.run(command, cwd=REMOTE_PROJECT_DIR, check=True)
    print(f'Finished {task_id} ({constraint_policy})')

    return {
        'task_id': task_id,
        'constraint_policy': constraint_policy,
        'shift_variant': shift_variant,
        'direction_share_variant': direction_share_variant,
        'cummax_variant': cummax_variant,
        'status': 'complete',
    }


def sample_task_ids(seed, split, number_of_tasks):
    """Sample task IDs reproducibly from a sorted ARC-AGI split."""
    dataset_path = project_root / 'dataset' / f'arc-agi_{split}_challenges.json'
    with dataset_path.open(encoding='utf-8') as file:
        all_task_ids = sorted(json.load(file).keys())

    if not 1 <= number_of_tasks <= len(all_task_ids):
        raise ValueError(
            f'number_of_tasks must be between 1 and {len(all_task_ids)}, '
            f'got {number_of_tasks}'
        )

    return random.Random(seed).sample(all_task_ids, number_of_tasks)


@app.local_entrypoint()
def main(
    seed: int = SEED,
    split: str = SPLIT,
    number_of_tasks: int = NUMBER_OF_TASKS,
    iterations: int = 1500,
    wandb_project: str = 'compressarc',
):
    """Sample tasks locally and execute both policies remotely."""
    task_ids = sample_task_ids(seed, split, number_of_tasks)
    is_full_ablation = (
        seed == SEED
        and split == SPLIT
        and number_of_tasks == NUMBER_OF_TASKS
        and iterations == 1500
    )
    execution_tag = 'full-ablation' if is_full_ablation else 'smoke-test'
    print(f'Sampled {number_of_tasks} task IDs with seed {seed}:')
    print(' '.join(task_ids))
    print(f'W&B execution tag: {execution_tag}')

    conditions = [
        {
            'task_id': task_id,
            'constraint_policy': constraint_policy,
            'split': split,
            'seed': seed,
            'number_of_tasks': number_of_tasks,
            'iterations': iterations,
            'wandb_project': wandb_project,
            'execution_tag': execution_tag,
        }
        for task_id in task_ids
        for constraint_policy in CONSTRAINT_POLICIES
    ]
    print(
        f'Launching {len(conditions)} conditions '
        f'({number_of_tasks} tasks x {len(CONSTRAINT_POLICIES)} policies) '
        'with at most 4 T4 containers at once.'
    )

    completed = []
    failures = []
    for result in run_condition.map(
        conditions,
        order_outputs=False,
        return_exceptions=True,
    ):
        if isinstance(result, BaseException):
            failures.append(result)
            print(f'Condition failed: {result!r}')
        else:
            completed.append(result)
            print(
                f"Completed {len(completed)}/{len(conditions)}: "
                f"{result['task_id']} ({result['constraint_policy']})"
            )

    print(f'Ablation finished: {len(completed)} complete, {len(failures)} failed.')
    if failures:
        raise RuntimeError(f'{len(failures)} Modal condition(s) failed')


def select_successful_baselines(runs, split):
    """Select finished original strict runs with an explicitly true pass@2."""
    selected = {}
    for run in runs:
        config = run.config
        if (run.state != 'finished' or config.get('split') != split
                or config.get('multitensor_constraints') != 'strict'
                or config.get('shift_variant', 'original') != 'original'
                or run.summary.get('pass_2_correct') is not True):
            continue
        task_id = config.get('task_name')
        if task_id:
            selected.setdefault(task_id, []).append(run.id)
    return {task: sorted(ids) for task, ids in sorted(selected.items())}


@app.function(timeout=24 * 60 * 60)
def run_convolution_queue(conditions):
    """Own the experiment remotely so local client disconnects cannot cancel it."""
    completed = []
    failures = []
    for index, condition in enumerate(conditions, start=1):
        print(
            f"Launching {index}/{len(conditions)}: {condition['task_id']} "
            f"({condition['shift_variant']})"
        )
        try:
            result = run_condition.remote(condition)
        except BaseException as error:
            failures.append({
                'task_id': condition['task_id'],
                'shift_variant': condition['shift_variant'],
                'error': repr(error),
            })
            print(f'Condition failed: {failures[-1]}')
        else:
            completed.append(result)
            print(f"Completed {index}/{len(conditions)}: {result}")
    print(f'Queue finished: {len(completed)} complete, {len(failures)} failed.')
    if failures:
        raise RuntimeError(f'{len(failures)} convolution condition(s) failed: {failures}')
    return completed


@app.local_entrypoint()
def convolution_ablation(
    wandb_project: str = 'grouparc',
    baseline_project: str = 'compressarc',
    wandb_entity: str = '',
    baseline_tag: str = 'full-ablation',
    split: str = 'training',
    iterations: int = 1500,
    dry_run: bool = False,
):
    """Run three shift variants only on successful original strict Modal tasks."""
    import wandb
    if iterations < 1:
        raise ValueError('iterations must be positive')
    api = wandb.Api()
    entity = wandb_entity or api.default_entity
    if not entity:
        raise ValueError('Supply --wandb-entity to identify the original project')
    source_project = f'{entity}/{baseline_project}'
    destination_project = f'{entity}/{wandb_project}'
    runs = api.runs(source_project, filters={'$and': [
        {'tags': 'modal'}, {'tags': 'strict-vs-relaxed'}, {'tags': baseline_tag},
    ]})
    selected = select_successful_baselines(runs, split)
    if not selected:
        raise ValueError(
            f'No successful original strict runs found in {source_project} '
            f'with tag {baseline_tag!r}'
        )
    with (project_root / 'dataset' / f'arc-agi_{split}_challenges.json').open() as file:
        known_tasks = json.load(file)
    if set(selected) - set(known_tasks):
        raise ValueError('Baseline selection contains tasks absent from the requested dataset')
    variants = ('original', 'projected_conv', 'pure_conv')
    conditions = [dict(
        task_id=task, constraint_policy='strict', shift_variant=variant,
        split=split, seed=SEED, number_of_tasks=len(selected), iterations=iterations,
        wandb_project=wandb_project, wandb_entity=entity,
        execution_tag='convolution-ablation', experiment='shift-convolution',
        source_run_ids=source_ids,
    ) for task, source_ids in selected.items() for variant in variants]
    manifest = dict(source_project=source_project,
                    destination_project=destination_project,
                    baseline_tag=baseline_tag,
                    selection_metric='pass_2_correct', selected_runs=selected,
                    conditions=conditions)
    manifest_path = project_root / 'convolution_ablation_manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(f'Selected {len(selected)} successful strict tasks from {source_project}: {", ".join(selected)}')
    print(f'Writing new runs to {destination_project}.')
    print(f'Manifest: {manifest_path}')
    print(f'{len(conditions)} runs: {len(selected)} tasks x 3 variants; one T4 container.')
    if dry_run:
        return
    run_convolution_queue.remote(conditions)
