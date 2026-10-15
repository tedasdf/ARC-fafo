# Four final L4 Slurm arrays

Submit from the ARC-fafo repository root after copying/synchronizing the current
`src` code, configs and scripts to the cluster. These scripts use the existing
`miniforge3` module and `SymARC` environment. Install `src/requirements.txt` in
that environment and authenticate with `wandb login` once.

```bash
module load miniforge3
mamba activate SymARC
python -m pip install -r src/requirements.txt
wandb login
sbatch src/slurm/final_original_20.sbatch
sbatch src/slurm/final_projected_conv_20.sbatch
sbatch src/slurm/final_projected_d4_20.sbatch
sbatch src/slurm/final_projected_lse_20.sbatch
```

Each file is one job array with 60 elements: 20 tasks x seeds 0, 1, 2. Each element
runs exactly one independent 1,500-step model. Index `3*t + s` selects task index
`t` in the config and seed index `s`. All four arrays together cover 240 runs.
`--array=0-59%1` permits one active GPU per array, so the four submissions permit
up to four active GPUs. Scheduling and your allocation determine availability.

Resources per element: `gpu` partition, `gpu:L4:1`, 4 CPUs, 16 GB RAM, 4 hours.
The L4 GRES spelling must match your cluster. Inspect your available GRES with
`sinfo -p gpu -o '%N %G'` and override with your actual token if needed, for
example `sbatch --gres=gpu:l4:1 src/slurm/final_original_20.sbatch`.
Supply your account/QoS flags on `sbatch` if your allocation requires them.
Scripts have LF line endings. No jobs have been submitted locally.

Logs use `slurm-<variant>-<array-job-id>_<index>.out` / `.err` at the repo root.
Final local models are under
`outputs/final_ablation/<variant>/<array-job-id>/<config-name>/seed_<seed>/<task>.pt`.
W&B is `arc_agi/compressarc`; names distinguish task, variant and seed. Each
completed run uploads its inference model artifact and final candidate table.

To retry failed elements, retain original indices, e.g.
`sbatch --array=5,17 src/slurm/final_projected_lse_20.sbatch`.
A resubmission starts fresh runs; checkpoints do not resume optimizer state.

Preview a single array element without submitting or using a GPU:

```bash
python src/run_training_timing.py --mode train --dry-run \
  --config src/config/models/final_original_20.yaml \
  --shard-count 60 --shard-index 0
```

Use `squeue -u "$USER"` to monitor queues and
`sacct -j ARRAY_JOB_ID --format=JobID,State,ExitCode` for completed elements.

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
