# Retained legacy tools

The remaining scripts here provide legacy inference, scoring, and Modal batch execution. They use the flat modules and dataset beside them, so run them from this folder with its requirements.txt installed. The current analysis package lives in ../src/compressarc/analysis/.

## Inference and scoring

| File | Purpose |
| --- | --- |
| solve_task.py | Worker for solving one task. It trains a fresh legacy model on CUDA and returns two attempts per test example through multiprocessing shared state. |
| scoring.py | Compares submission.json with the configured ground-truth solutions file and prints aggregate and per-task scores. |

## Batch execution

| File | Purpose |
| --- | --- |
| modal_runner.py | Runs sampled task experiments on Modal using src/train.py for strict and relaxed multitensor constraints. |

## Legacy analysis data

The analysis package modules list_solved_puzzles and plot_accuracy consume prediction histories written by the legacy solution_selection.py. They preserve compatibility with existing NPZ files and do not read W&B runs. The new trainer does not currently create those histories.

## Legacy runtime dependencies

The retained scripts still import the legacy model and training implementation directly. Keep these beside the tools unless the tools are refactored to use src/:

- arc_compressor.py, layers.py, initializers.py, and multitensor_systems.py
- preprocessing.py, train.py, and solution_selection.py
- The split data under dataset/ and the legacy Python dependencies in requirements.txt

Legacy train.py and solve_task.py use the shared visualizer from ../src/compressarc/analysis/visualization.py.

## Migration boundary

The new model and standard trainer live under ../src/. The current analysis package also lives under ../src/compressarc/analysis/. The new trainer does not call solve_task.py or modal_runner.py.