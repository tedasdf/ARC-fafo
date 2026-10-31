# Optional per-task CUDA Graph training

Enable with `--set training.cuda_graph=true`. Eager training remains the default.
Defaults: `cuda_graph_start=100`, `cuda_graph_warmup=3`. The start must be at least
`annealing_steps`, and the graph is constructed independently for each task/model.
For a 1500-update task this means 100 eager updates and 1400 graph-backed updates.
Capture preparation never calls Adam and does not count as an update. Its RNG state
is restored before actual replay. Parameter shapes and memory addresses must stay fixed.

The graph includes gradient clearing, forward, objective and backward. Adam, diagnostics,
CPU metric/output transfers, candidate decoding/ranking and W&B logging stay eager.
Only parameters that actually receive gradients get static gradient buffers; unused
parameters retain `grad=None`. RNG state is replay-aware, rather than freezing latent noise.
The loss coefficients are captured only after annealing has finished. The latent decoder's
two tensor-valued scalar constants were replaced by equivalent Python scalars to avoid
synchronous CPU-to-GPU copies during capture.

Supported timing: `training.step_timing=off` or `wall`. Operator profiling, sharing-stage
timing, layer timing and benchmark mode are rejected rather than emitting misleading phase
times. Capture failure raises with its underlying error; no optimizer update occurs for that
step and there is no silent fallback. Short runs ending before capture remain eager and
report `captured=false`.

Example on a CUDA machine, from the repository root:

```bash
python src/train.py --config experiments/seed0_d4conv_retest/configs/projected_conv_d4.json \
  --task 00576224 --set training.split=evaluation --set training.cuda_graph=true \
  --set training.step_timing=wall --set logging.wandb=false --set logging.save_model=false
```

Outputs include `<task>.cuda_graph.json` with capture seconds and exact eager/replayed/update
counts. W&B-enabled runs receive a CUDA Graph tag and `cuda_graph/*` summary fields.
The existing candidate and accuracy metrics retain their normal meaning.

## Validation and GPU gate

Local CPU checks: **39 passed, 5 CUDA-dependent tests skipped**. These cover default-eager
training/checkpoint behavior, configuration guards, optimizer/timing behavior and exact
scalar-cleanup output/gradient/RNG parity in float32/float64. CUDA capture has not been
executed locally (CPU-only Torch), so this is an opt-in implementation pending GPU validation.

Run `PYTHONPATH=src python src/cuda_graph_preflight.py` on a CUDA/Triton machine first.
It compares a small actual D4 model against eager updates after annealing, including
outputs, gradients, Adam state, inactive gradients, update counts and CUDA RNG advancement.
If the preflight fails, retain the error and do not use the graph path for a full submission.

The provided Kaggle benchmark notebook applies the source update in its writable copy,
runs that preflight, then measures eager versus graph on one/four GPU workers with matched
settings. It keeps original vectorized direction sharing in both arms, so the graph result
is independent of the separate optional `vectorized_packed` optimization. Two repeats,
180 real updates per worker; discard updates 0–119, measure 120–179. Capture cost is recorded
separately. These one-task steady results do not establish whole-dataset runtime or identical
full-training accuracy. No package versions, external source datasets, or already-running
notebooks are modified by the benchmark.
