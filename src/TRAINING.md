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

It calls `train.main` for the five real tasks listed in the launcher and each
cummax variant (`primitives`, `d4`, `optimised`). Pass the same exploratory YAML
with `--config`; standard `--set` overrides also work. Model initialization,
actual task preprocessing, loss, and Adam update are the existing trainer code.

Every task/variant gets two fresh, equally seeded passes and two W&B runs:

- **clean**: wall-timed setup, first step, 10 warmup updates (including the first),
  then a synchronized block of 30 complete updates. Only this pass reports
  complete-step timing and peak memory. Peaks reset immediately before the block.
- **operations**: the same initialization and 10 + 30 updates, with CUDA events
  around forward, loss, backward, optimizer, and metric transfers. Gradient
  clearing remains inside complete updates without its own metric. Profiling
  results do not need to sum to clean block wall time.

Thus the default is 40 updates per pass, 80 per task/variant, 30 W&B runs total.
W&B initialization and summary writes happen outside timed regions. Benchmark
mode skips predictions/images, PCA, progress updates, and checkpoints; scalar
transfers already inside `take_step` remain included. The loss-phase KL-component
transfers and final metric/output transfers are assigned distinct boundaries.
Metrics use the definitions in `training_metrics_spec.py`; they are written to
W&B run summaries, and task/model settings remain in the existing W&B config.

For one task/variant or a different measured block:

```bash
python src/run_training_timing.py --task 694f12f3 --variant optimised --measured-iterations 50
```

Use `--clean-only` to skip profiling. CPU smoke runs use host-clock operation
spans and omit GPU memory; their W&B config identifies the CPU method. They do
not provide GPU timing validation. To avoid contacting W&B during a smoke run,
pass `--set logging.wandb=false`. Normal `train.py` defaults to `--mode train`.
The benchmark flag can also be invoked directly for a single pass:

```bash
python src/train.py --mode benchmark --task 694f12f3 --benchmark-pass clean --set logging.wandb=true
```
