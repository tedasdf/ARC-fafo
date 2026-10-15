# Four task-group L4 Slurm arrays

Submit from the updated ARC-fafo repository root. The scripts use the existing
`miniforge3` module and `SymARC` environment; install `src/requirements.txt` and
log into W&B once before submitting.

```bash
sbatch src/slurm/final_tasks_01.sbatch
sbatch src/slurm/final_tasks_02.sbatch
sbatch src/slurm/final_tasks_03.sbatch
sbatch src/slurm/final_tasks_04.sbatch
```

Each array owns five distinct tasks and covers all four variants and three
seeds for each task: 5 x 4 x 3 = 60 elements. Across the four arrays: 240 models.
Each element trains one independent 1,500-update model, retaining the four-hour
limit. `--array=0-59%1` allows one active GPU per array (up to four across the
four submissions). Array index = 12 * local_task_index + 3 * variant_index + seed_index.
Variant order: original, projected convolution, projected D4, projected LSE.
Seeds: 0, 1, 2. The experiment and metric definitions are unchanged.

| Script | Task IDs |
|---|---|
| `final_tasks_01.sbatch` | `3eda0437`, `0dfd9992`, `95990924`, `7f4411dc`, `6f8cd79b` |
| `final_tasks_02.sbatch` | `445eab21`, `3bdb4ada`, `36fdfd69`, `d43fd935`, `1c786137` |
| `final_tasks_03.sbatch` | `1bfc4729`, `3aa6fb7a`, `6cdd2623`, `74dd1130`, `f8c80d96` |
| `final_tasks_04.sbatch` | `623ea044`, `d406998b`, `6d75e8bb`, `d9fac9be`, `97999447` |

Each element requests `--partition=gpu --gres=gpu:L4:1`, four CPUs and 16 GB RAM.
Use your cluster's actual GRES token and account/QoS flags if different.
For available GPU names: `sinfo -p gpu -o '%N %G'`.

Logs: `slurm-tasks_<group>-<array-job-id>_<index>.out` / `.err`.
Checkpoints:
`outputs/final_ablation/tasks_<group>/<array-job-id>/<config-name>/seed_<seed>/<task>.pt`.
W&B project: `arc_agi/compressarc-final-ablation`; task, variant and seed remain
in run names/tags. Final model uploads and candidate tables remain enabled.

To retry only failed elements, keep their original indices, for example:
`sbatch --array=5,17 src/slurm/final_tasks_01.sbatch`.
Resubmissions start fresh runs, not optimizer-state resumes.

The previous per-variant array files have been replaced by these task-group
files. Do not also submit old downloaded per-variant scripts; they would
repeat the same experiment. No jobs have been submitted by these changes.

Monitor with `squeue -u "$USER"` and
`sacct -j ARRAY_JOB_ID --format=JobID,State,ExitCode`.

## Zero-step smoke test

Submit one short L4 allocation to check all four variants and seeds 0/1/2 on
small selected task `74dd1130` (12 checks total):

```bash
sbatch src/slurm/test_zero_steps.sh
```

Inside an existing L4 allocation, run `bash src/slurm/test_zero_steps.sh`.
It constructs each real model and performs one finite, no-gradient forward.
There is no optimizer, training, W&B run or checkpoint upload. This tests one
sampled task, not all 240 full training jobs.

Use `--init-only` for construction with no forward, or `--seed 0` for four checks.
If your environment is already active, set `ARC_SKIP_ENV=1`. For CPU validation:

```bash
ARC_SKIP_ENV=1 bash src/slurm/test_zero_steps.sh --device cpu --seed 0
```

Normal training deliberately rejects `training.iterations=0`; this dedicated
script tests setup without weakening that guard. GPU GRES/account/QoS rules from
the submission section also apply to this script.

Repository discovery checks the current directory first, then Slurm's submit
directory and the script location. To diagnose paths without loading modules
or using GPUs, run `bash src/slurm/test_zero_steps.sh --check-root`.

The smoke test sets PyTorch's global default device explicitly and verifies
that task tensors, parameters and buffers are on the chosen device before forward.

## Run one task/variant/seed first

```bash
sbatch src/slurm/final_single.sbatch
```

Defaults: task `74dd1130`, original control, seed 0, 1,500 updates, one L4.
This is a real training run with normal W&B logging and final model upload.
No array or seed sweep is used. Override its three positional arguments:

```bash
sbatch src/slurm/final_single.sbatch 74dd1130 projected-conv 0
```

Variants: `original`, `projected-conv`, `projected-d4`, `projected-lse`.
Checkpoints use `outputs/final_single/<job-id>/<config-name>/seed_<seed>/`.
This pilot is additional to the 240-run arrays; submitting all arrays later
will repeat its task/variant/seed combination.
