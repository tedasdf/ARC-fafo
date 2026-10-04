"""Run a seeded strict-vs-relaxed ablation with the src trainer on Modal."""

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
    .pip_install_from_requirements(str(project_root.parent / 'src' / 'requirements.txt'))
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
            '__pycache__/**',
            'wandb/**',
            'notebook/**',
            'results_for_the_blog_post/**',
            '*.pyc',
        ],
    )
    .add_local_dir(
        project_root.parent / "src",
        remote_path=f"{REMOTE_PROJECT_DIR}/src",
        ignore=["**/__pycache__/**", "*.pyc"],
    )
)
app = modal.App('compressarc-strict-relaxed-ablation')


@app.function(
    image=image,
    gpu='T4',
    secrets=[modal.Secret.from_name('wandb-secret', required_keys=['WANDB_API_KEY'])],
    timeout=2 * 60 * 60,
    max_containers=4,
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

    wandb_tags = (
        f"[random-{number_of_tasks},sample-seed-{seed},"
        f"strict-vs-relaxed,modal,{execution_tag}]"
    )
    command = [
        sys.executable,
        "src/train.py",
        "--task", task_id,
        "--set", f"training.split={split}",
        "--set", f"training.iterations={iterations}",
        "--set", f"training.seed={seed}",
        "--set", "logging.wandb=true",
        "--set", "logging.latent_pca=true",
        "--set", f"logging.wandb_project={wandb_project}",
        "--set", f"logging.wandb_tags={wandb_tags}",
        "--set", f"model.multitensor_constraints={constraint_policy}",
    ]

    print(f'Starting {task_id} ({constraint_policy})')
    print(shlex.join(command))
    subprocess.run(command, cwd=REMOTE_PROJECT_DIR, check=True)
    print(f'Finished {task_id} ({constraint_policy})')

    return {
        'task_id': task_id,
        'constraint_policy': constraint_policy,
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
