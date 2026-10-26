# Training

Install dependencies from the repository root:

```bash
python -m pip install -r src/requirements.txt
```

Train one task, or omit `--task` to train the configured split:

```bash
python src/train.py --task 50cb2852 --set training.iterations=1500
```

Defaults live in `src/config/default.yaml`. Use `--config <yaml>` and repeat
`--set KEY=VALUE` to override them. Select strict or relaxed preprocessing with
`model.multitensor_constraints`.

The solution tracker updates every optimizer step. Console output shows progress,
final loss, both guesses, and correctness when reference solutions are available.
W&B logs `train/reconstruction_error` (unweighted reconstruction CE objective)
and `train/total_KL` every step, plus `predictions/top_1_correct` and
`predictions/pass_2_correct` when reference answers exist.

The solution tracker updates every step. Every 50 updates, W&B logs `candidates/cumulative_unique_count`,
`candidates/cumulative_correct_count`, `candidates/best_correct_rank`,
`candidates/best_correct_score`, and `candidates/top_ranked_score`.
Correctness fields require reference answers. Before a correct candidate exists,
its rank and score are null (not zero). These rank and score fields use the current ranking at each logged step, not
historical best values. A correct candidate can fall from rank 1 to rank 2 or
lower. Rank 1 is the highest accumulated
selection score; equal scores follow first appearance. Candidates are exact
full tuples of test-output grids, deduplicated across the run. Correct candidate
count therefore normally stays at 0 or 1 for a task with one accepted answer.

`candidates/first_correct_generation_step` is stored in the summary immediately
on first generation, using 1-based steps. If never generated, it is absent.
At normal completion, `candidates/all_unique` contains one row per unique output
hash with exactly four columns: `hash`, `first_step` (1-based), `occurrences`,
and `is_correct` (exact match, null when reference unavailable). Occurrences
count a candidate at most once per step when raw and EMA agree, and count
reappearances on later steps. Final candidate metrics remain in the summary.

```bash
wandb login
python src/train.py --task 50cb2852 --set logging.wandb=true --set logging.wandb_entity=arc_agi
```

The default project is `compressarc`. Configure the account, project and mode with
`logging.wandb_entity`, `logging.wandb_project` and `logging.wandb_mode`.
Logging is disabled by default.

## Local checkpoints

```bash
python src/train.py --task 50cb2852 --save-checkpoints --output-dir outputs/training
```

This saves final weights, model configuration, task information, and scalar
metrics in a local `.pt` file. It is a final model snapshot, not a resumable
optimizer checkpoint.

## Modal

Install and authenticate Modal, then use the same trainer:

```bash
pip install modal
modal setup
python src/train.py --backend modal --task 50cb2852 --save-checkpoints
```

The default GPU is T4; select another with `--modal-gpu`. Saved checkpoints are
returned to the local output directory. When W&B is enabled, configure a Modal
secret named `wandb-secret` containing `WANDB_API_KEY`.

## Benchmarks

```bash
python src/train.py --mode benchmark --task 694f12f3 --benchmark-pass clean
python src/train.py --mode benchmark --task 694f12f3 --benchmark-pass operations
```

Both passes use the existing model, objective, and optimizer. Clean reports setup,
complete-step timing and memory; operations profiles phases. Results print as JSON
to the console. Defaults are 10 warmup and 30 measured updates. Benchmark mode does
not save model checkpoints.

The batch launcher supports configured variants and task sharding:

```bash
python src/run_training_timing.py --set training.device=cuda:0
python src/run_training_timing.py --mode train --save-checkpoints
```

Use `--help` on the trainer or launcher for configuration and sharding options.

## Per-run summaries

With W&B enabled, each task records `run/wall_clock_seconds`, `run/status`
(`running`, `finished`, `failed`, or `interrupted`), and on CUDA,
`run/peak_gpu_allocated_mib` and `run/peak_gpu_reserved_mib`.
Wall time covers model/optimizer setup, training, tracking and local checkpoint
saving; it excludes dataset loading, W&B initialization and final upload/finish.
GPU peaks reset before constructing each task model. They measure PyTorch-managed
allocations/reservations, not total device memory. CPU runs omit GPU fields.
Caught failures finish W&B with a nonzero exit code. Forced termination cannot
write final summaries and is reflected by W&B's own run state.

## Kernel and numerical diagnostics

With W&B enabled, trainable directional cores are monitored individually by
layer type, depth and parameter name (including the direction-share `theta`).
Primitive-only paths have no trainable core kernels and omit these measurements.
The parameter tensor count is stored as `kernels/parameter_tensor_count`.

Every 50 updates (50, 100, ...), collection and logging include:
- `kernels/<layer>/depth_<n>/<parameter>/distance_from_initialization`: absolute L2 distance to the initialization snapshot.
- `kernels/<layer>/depth_<n>/<parameter>/gradient_norm`: L2 gradient norm after backward, before Adam. Missing gradients are null.
- `kernels/<layer>/depth_<n>/<parameter>/parameter_norm`: L2 parameter norm after Adam.

Every training update checks raw model outputs and KL contributions, loss terms,
all available parameter gradients before Adam, and all parameters after Adam.
Nonfinite events emit `numerics/phase`, `numerics/nan_count` and
`numerics/inf_count` with `train_step`. Finite observations emit no event.
Final summaries always include `numerics/nonfinite_detected`, `nan_detected`,
`inf_detected`, `nonfinite_event_count`, and `nonfinite_step_count` under the
`numerics/` prefix. `first_nonfinite_step` is 1-based and present only if detected.
Event counts count observed phases, so the same nonfinite value propagated
through several phases can generate multiple events in one step. Checks are
observational; they do not skip optimizer updates or sanitize values.

## Final model upload

Each completed training run with W&B enabled automatically saves a final `.pt`
checkpoint under `--output-dir` and attaches it as a W&B artifact of type `model`,
named `model-<run-id>` with aliases `final` and `latest`. No `--save-checkpoints`
flag is needed for W&B runs; that flag still controls local-only runs.
The versioned inference file contains final parameter values and layer buffers,
model configuration, task inputs/training examples, color/shape metadata,
training seed/iteration count, scalar losses, and Python/NumPy/PyTorch runtime
versions. Selection state includes its EMA tensors, candidate scores and records,
selected guesses, first-correct step, EMA decay, uncertainty weight, early-step
penalty/cutoff, EMA penalty, and tie/crop rules. CPU/CUDA Torch, NumPy and Python
RNG states are included because decoder inference is stochastic.

Logging configuration, credentials and optimizer state are excluded. This is an
inference checkpoint, not a resumable training checkpoint. Benchmark mode does
not upload a final model. Bitwise inference reproduction is tested on the same
CPU/runtime; CUDA kernels and different hardware/software can differ.

```python
from compressarc.train.checkpoint import load_inference_checkpoint

saved = load_inference_checkpoint("outputs/training/50cb2852.pt", device="cpu")
print(saved.tracker.solution_most_frequent)  # Original selected final guess.
print(saved.tracker.solution_second_most_frequent)
# saved.model.forward() reproduces the next sampled model forward on the same
# runtime/device, from the RNG state saved after training.
```

The saved model weights are after the final optimizer update; the stored selected
guesses include the last forward pass made before that update. These are distinct
states, both retained. Use the same code/backend implementation as the saved
configuration, with `src` importable, to reconstruct the model.



## Per-step latency and bottleneck profiling

`training.step_timing` accepts `off` (the default), `wall`, or `phases`. The D4-conv pilot now enables `wall` in its config for future launches. Already-running processes are not changed.

- `wall` records total step time, including the model update, output transfers, candidate tracking, diagnostics and normal scalar logging.
- `phases` additionally records host wall durations for forward, loss, backward, optimizer, metric extraction, progress display, post-step diagnostics, candidate tracking and logging. On CUDA it also records stream elapsed times with events. Only step boundaries synchronize; phase boundaries do not.
- CUDA stream elapsed includes CPU-dispatch gaps and is not GPU kernel-active time. Host and stream times overlap; never add them together.
- Numerics checks inside take_step are included in their enclosing phase; the separate diagnostics phase measures diagnostics.after_step only.

Every measured step is written to `<output-dir>/<task>.step_timing.jsonl` and, when enabled, W&B under `timing/*` with train_step as the x-axis. Summary statistics go to `<task>.step_timing.summary.json`. The summary excludes the first ten completed measurements from steady statistics when more than ten are available. First-step and periodic 50-step overhead remain visible in the per-step file. Timing-export overhead is excluded from the step metric and recorded separately, including an all-step mean with export overhead included. Data loading, model construction, final checkpoints/artifacts and final candidate-table uploads are outside the per-step timing.

For a short D4-only performance experiment in an allocated GPU session:

```bash
python src/train.py --config experiments/d4_conv_pilot20/configs/projected_conv_d4.json --task 00576224 --set training.split=evaluation --set training.device=cuda:0 --set training.iterations=100 --set training.step_timing=phases --set experiment.name=projected_conv_d4_latency_profile --output-dir outputs/d4_conv_latency_profile
```

The public ARC-AGI-1 evaluation fixture above shares the same task ID/inputs as a Kaggle public task; use the actual Kaggle dataset/runtime for a deployment-representative benchmark. For a full existing experiment worker, add `--step-timing phases` or `--step-timing wall` after `--index N`. Retest production latency with wall mode after finding bottlenecks. Phase instrumentation adds overhead, and CPU tests cannot establish a 0.15-second GPU target. This does not change the model operators or training objective.
