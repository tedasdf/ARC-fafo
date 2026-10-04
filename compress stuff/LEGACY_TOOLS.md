# Retained legacy tools

The scripts in this folder remain the legacy inference, scoring, and analysis workflow. They are intentionally staying here rather than being moved into the new `src/` package. They use the flat modules and dataset beside them, so run them from this folder with its `requirements.txt` installed.

## Inference and scoring

| File | Purpose |
| --- | --- |
| `solve_task.py` | Worker for solving one task. It loads a task, trains a fresh model on CUDA until its iteration limit or deadline, and returns two attempts per test example through multiprocessing shared state. It also reports peak GPU memory and sends failures to an error queue. It is a worker function for custom multiprocessing callers; it does not create a submission file by itself. |
| `scoring.py` | Compares `submission.json` with the configured ground-truth solutions file and prints aggregate and per-task scores. The filenames are currently set in the script. |

## Batch execution

| File | Purpose |
| --- | --- |
| `modal_runner.py` | Runs sampled task experiments on Modal. The current experiment invokes `analyze_example_wandb.py` for strict and relaxed multitensor constraints. |

## Analysis and visualization

| File | Purpose |
| --- | --- |
| `analyze_example.py` | Trains one task locally and saves training plots and latent PCA visualizations. |
| `analyze_example_wandb.py` | Runs one or more tasks and optionally logs puzzle images, training metrics, predictions, final guesses, and latent PCA to Weights & Biases. |
| `list_solved_puzzles.py` | Reads saved prediction-contribution histories and reports which tasks had their solution ranked within a chosen guess count at a selected iteration. |
| `plot_accuracy.py` | Computes and plots pass-at-*n* accuracy from saved prediction histories. |
| `plot_problems.py` | Creates visualizations of the ARC puzzles in a dataset split. |
| `visualization.py` | Shared plotting functions used by the legacy analysis scripts. |

`list_solved_puzzles.py` and `plot_accuracy.py` consume prediction histories written by the legacy `solution_selection.py`; they do not read W&B runs.

## Legacy runtime dependencies

The retained scripts still import the legacy model and training implementation directly. Keep these beside the tools unless the tools are refactored to use `src/`:

- `arc_compressor.py`, `layers.py`, `initializers.py`, and `multitensor_systems.py`
- `preprocessing.py`, `train.py`, `solution_selection.py`, and `visualization.py`
- The split data under `dataset/` and the legacy Python dependencies in `requirements.txt`
## Migration boundary

The new model and standard trainer live under `../src/`. The tools listed here remain tied to the legacy flat modules in this folder. The new trainer does not call `solve_task.py`, `modal_runner.py`, or the NPZ-based analysis utilities.


