# Retained legacy tools

The remaining scripts here provide legacy inference and Modal batch execution. They use the flat modules and dataset beside them, so run them from this folder with its requirements.txt installed. The current analysis package lives in ../src/compressarc/analysis/.

## Inference

| File | Purpose |
| --- | --- |
| ../src/compressarc/analysis/solve_task.py | Worker for solving one task. It trains a fresh legacy model on CUDA and returns two attempts per test example through multiprocessing shared state. |

Submission scoring lives in ../src/compressarc/analysis/scoring.py. It defaults to ../src/dataset/arc-agi_training_solutions.json; pass --solutions to score against another split.

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
- The split data under dataset/ for the deferred solver. Shared Python dependencies are maintained in ../src/requirements.txt; the local requirements.txt forwards to that file.

Legacy train.py uses the shared visualizer from ../src/compressarc/analysis/visualization.py. The analysis solve_task module adds the legacy folder to its import path to reuse the original model and training modules.

## Deferred: solve_task migration

Status: deferred at the user's request. Moving `solve_task.py` into `src/compressarc/analysis/` has not completed its migration; its current runtime path needs repair before use.

- The worker still constructs the legacy `arc_compressor.ARCCompressor`, uses `solution_selection.Logger`, and reads `compress stuff/dataset`.
- It inserts the legacy folder into `sys.path`, then inserts `src` ahead of it. In a fresh process, `import train` and `import preprocessing` resolve to the new source modules while the remaining imports select legacy modules.
- Its `train.take_step(..., train_history_logger)` call uses the old interface. The new function expects training configuration in that position, so this mixed import path is incompatible.

Follow-up: migrate the worker to explicit imports of the new model, objective, preprocessing, and prediction tracker; preserve its time limit, CUDA worker setup, two-attempt output, and multiprocessing result/error reporting. Validate a single-task worker run before removing the legacy dependencies. Avoid relying on `sys.path` ordering to choose between old and new modules.

## Migration boundary

The new model and standard trainer live under ../src/. The current analysis package also lives under ../src/compressarc/analysis/. The new trainer does not call solve_task.py or modal_runner.py.