# Training

Install the shared training dependencies from the repository root:

~~~powershell
python -m pip install -r src/requirements.txt
~~~

The standard trainer, Modal image, and scoring defaults use files under `src`. The deferred `compressarc.analysis.solve_task` worker still needs the legacy folder.

From the repository root, run the standard trainer locally:

~~~powershell
python src/train.py --task <task-id> --set training.iterations=500
~~~

Omit --task to train every task in the configured split. The defaults are in src/config/default.yaml; use --config for another YAML file and repeat --set KEY=VALUE for one-off overrides.

## Run on Modal

Install the Modal CLI in the local Python environment and authenticate once:

~~~powershell
pip install modal
modal setup
~~~

Then use the same trainer and config with the Modal backend:

~~~powershell
python src/train.py --backend modal --task <task-id> --set training.iterations=500
~~~

Omit --task to run the full configured split remotely. Modal uses a T4 GPU by default; choose another supported GPU with --modal-gpu, for example --modal-gpu A100.

To save checkpoints, add --save-checkpoints. The remote run returns checkpoint files to the local --output-dir (default: outputs/training).

Use model.multitensor_constraints=strict (the default) or model.multitensor_constraints=relaxed to select the preprocessing policy. When W&B is enabled in the config, the Modal run expects a Modal secret named wandb-secret containing WANDB_API_KEY:

~~~powershell
modal secret create wandb-secret WANDB_API_KEY=<your-wandb-api-key>
~~~

The Modal backend runs the resolved configuration remotely; local execution remains the default.

## Timing through the existing trainer

Run the thin launcher from the repository root on your single GPU:

```bash
python src/run_training_timing.py --set training.device=cuda:0
```

It calls `train.main` for the five real tasks listed in the launcher and each of the five model YAMLs in `src/config/models`. Pass the same exploratory YAML
with `--config`; standard `--set` overrides also work. Model initialization,
actual task preprocessing, loss, and Adam update are the existing trainer code.

Every task/configuration gets two fresh, equally seeded passes and two W&B runs:

- **clean**: wall-timed setup, first step, 10 warmup updates (including the first),
  then a synchronized block of 30 complete updates. Only this pass reports
  complete-step timing and peak memory. Peaks reset immediately before the block.
- **operations**: the same initialization and 10 + 30 updates, with CUDA events
  around forward, loss, backward, optimizer, and metric transfers. Gradient
  clearing remains inside complete updates without its own metric. Profiling
  results do not need to sum to clean block wall time.

Thus the default is 40 updates per pass, 80 per task/configuration, 50 W&B runs total.
W&B initialization and summary writes happen outside timed regions. Benchmark
mode skips predictions/images, PCA, progress updates, and checkpoints; scalar
transfers already inside `take_step` remain included. The loss-phase KL-component
transfers and final metric/output transfers are assigned distinct boundaries.
Metrics use the definitions in `training_metrics_spec.py`. The operation pass
buffers one timing record per measured step, then replays them in step order
to W&B after the timed pass. Charts use `train_step` (10-39 by default); run
summaries retain measured means. Clean setup, block time, and memory peaks
remain summary metrics. The clean run also records `timing/first_step_ms` at
`train_step=0` in history, buffered until measurement finishes. Task/model
settings remain in the existing W&B config.

For one task/variant or a different measured block:

```bash
python src/run_training_timing.py --task 694f12f3 --config src/config/models/all_projected.yaml --measured-iterations 50
```

Use `--clean-only` to skip profiling. CPU smoke runs use host-clock operation
spans and omit GPU memory; their W&B config identifies the CPU method. They do
not provide GPU timing validation. To avoid contacting W&B during a smoke run,
pass `--set logging.wandb=false`. Normal `train.py` defaults to `--mode train`.
The benchmark flag can also be invoked directly for a single pass:

```bash
python src/train.py --mode benchmark --task 694f12f3 --benchmark-pass clean --set logging.wandb=true
```


## Model YAMLs for timing and exploratory runs

| YAML in src/config/models | Shift | Direction-share | Cummax |
|---|---|---|---|
| original.yaml | primitives | primitives | primitives |
| projected_shift.yaml | tied_conv | primitives | primitives |
| projected_direction_share.yaml | primitives | d4 | primitives |
| projected_cummax_lse.yaml | primitives | primitives | optimised LSE |
| all_projected.yaml | tied_conv | d4 | optimised LSE |

Each YAML inherits other model and training settings from `config/default.yaml`.
The projected D4 implementation migrates the pilot's
`x + W2 D4(W1 normalize(x))`, using 8-channel projections, XY symmetrization,
and one 10-orbit D4 mixer per depth. Legacy directional pair weights consume
their original initialization sequence but are removed from the optimizer.
The `d4` factory selection uses this projected wrapper in
`direction_share/morphological.py`; there is no separate projected implementation file.

The launcher no longer has a variant flag or silently overrides model choices.
Pass `--config` repeatedly to choose several YAMLs; omit it to run all five.
Explicit `--set` overrides still apply. W&B records each configuration name,
source YAML, and the actual resolved model settings. Use the same YAML for pilots:

```bash
python src/train.py --config src/config/models/all_projected.yaml --task 694f12f3 --set logging.wandb=true
```

These preserve the migrated layer architecture; pilot training settings such as
seed 42, training length, and early stopping are not implicitly imported. Set
those explicitly if needed. The LSE YAML uses the optimized PyTorch LSE backend.


## Explicit optimisation and Triton cummax configs

`projected_cummax_optimisation.yaml` selects the same optimized PyTorch backend
as the existing `projected_cummax_lse.yaml`. `projected_cummax_triton.yaml` selects
a full LSE wrapper with Triton cardinal and diagonal forward/backward. Both replace only cummax and preserve projections/masks/residuals.
They are additional explicit configs; the launcher's original five defaults
are unchanged.

```bash
python src/run_training_timing.py --config src/config/models/projected_cummax_optimisation.yaml --config src/config/models/projected_cummax_triton.yaml --set training.device=cuda:0
```

Triton requires Linux CUDA and float32. Install `src/requirements.txt` on the L4
and verify the GPU parity tests before comparing performance:

```bash
cd src
python -m pytest tests/correctness/layers/lse/test_triton_lse.py -q
```

Both canonical scans use Triton; direction rotations, projections, masking,
and residual operations remain standard PyTorch wrapper logic. Optional imports
are lazy so ordinary CPU/PyTorch runs do not require Triton.


## Comparing tied-convolution implementations

The existing `projected_shift.yaml` uses grouped PyTorch `conv2d`.
`projected_shift_unfold.yaml` uses explicit `F.unfold` and weighted patch
multiplication. `projected_shift_triton.yaml` uses a direct 3x3 depthwise Triton
kernel and input/filter backward kernels. All retain the same two canonical
learned 3x3 kernels, direction rotations, parameter/state-dict layout, masks,
projections, normalization and residuals. No matrix fusion changes the model.

```bash
python src/run_training_timing.py --config src/config/models/projected_shift.yaml --config src/config/models/projected_shift_unfold.yaml --config src/config/models/projected_shift_triton.yaml --set training.device=cuda:0
```

These explicit configs do not change the launcher's five defaults. Verify the
optional GPU kernel before comparing full-step timings, from `src`:

```bash
python -m pytest tests/correctness/layers/shift/test_shift.py -q
```

## Full training on the original eight pilot tasks

The final_pilot_8.yaml config selects conv2d projected shift, projected D4,
and Triton cardinal/diagonal LSE. It declares the original eight task IDs,
1500 updates per task, seed 42, and strict constraints. Other model and
optimizer defaults stay shared. Full training uses the existing trainer and
W&B training metrics/predictions rather than clean/profile benchmark passes.

From the repository root on the L4:

```bash
python src/run_training_timing.py --mode train --config src/config/models/final_pilot_8.yaml
```

Add --dry-run to inspect all eight planned calls without training. The launcher
reads experiment.task_ids when --task is omitted; explicit --task selections
still take precedence. Benchmark mode remains the launcher default. The pilot
run is fixed at 1500 updates per task; legacy early stopping is not enabled.
Each task gets a fresh model/optimizer and seed 42, trained sequentially on
cuda:0, with a separate W&B run in arc_agi/compressarc.
