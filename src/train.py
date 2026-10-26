"""Train ARCCompressor tasks locally or on Modal with optional correctness logging."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

import preprocessing
from config import load_config
from compressarc.model.model import ARCCompressor
from compressarc.train import take_step
from compressarc.train.metrics import SolutionTracker
from compressarc.train.checkpoint import save_inference_checkpoint
from compressarc.train.diagnostics import TrainingDiagnostics
from compressarc.train.step_timing import StepTimer
from compressarc.train.logging import initialize_wandb, log_training_metrics, log_final_candidates, log_model_checkpoint, RunSummary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="See TRAINING.md for Modal setup and examples.",
    )
    parser.add_argument("--mode", choices=("train", "benchmark"), default="train")
    parser.add_argument("--benchmark-pass", choices=("clean", "operations"), default="clean")
    parser.add_argument("--warmup-iterations", type=int, default=10)
    parser.add_argument("--measured-iterations", type=int, default=30)
    parser.add_argument("--config", type=Path, help="YAML config merged over config/default.yaml")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a YAML value; may be repeated, e.g. training.iterations=500",
    )
    parser.add_argument("--task", help="ARC task ID; omit to train every task in the split")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "outputs" / "training",
        help="Directory used when --save-checkpoints is enabled",
    )
    parser.add_argument(
        "--backend",
        choices=("local", "modal"),
        default="local",
        help="Run training locally or submit the same configured run to Modal",
    )
    parser.add_argument(
        "--modal-gpu",
        default="T4",
        help="GPU type for the Modal backend (default: T4)",
    )
    parser.add_argument("--save-checkpoints", action="store_true",
                        help="Save final .pt checkpoints locally")
    return parser.parse_args(argv)


def merge_cli_config(args):
    return load_config(args.config, args.overrides)


def main(argv=None):
    args = parse_args() if argv is None else parse_args(argv)
    mode = getattr(args, "mode", "train")
    config = merge_cli_config(args)
    training_config = config.training
    if mode == "benchmark":
        if not args.task or args.backend != "local":
            raise ValueError("Benchmark mode requires --task and the local backend")
        if args.warmup_iterations < 1 or args.measured_iterations < 1:
            raise ValueError("Warmup and measured iteration counts must be positive")
        training_config.iterations = args.warmup_iterations + args.measured_iterations
        if config.model.layer_timing:
            raise ValueError("Disable model.layer_timing for clean/profile pass separation")
    if training_config.iterations < 1 or training_config.learning_rate <= 0 or training_config.annealing_steps < 1:
        raise ValueError("iterations, learning_rate, and annealing_steps must be positive")
    if args.backend == "modal":
        from compressarc.train.modal_launcher import launch_modal

        launch_modal(config, args)
        return
    np.random.seed(training_config.seed)
    torch.manual_seed(training_config.seed)
    device = training_config.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device_object = torch.device(device)
    if device_object.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but no CUDA device is available")
    if device_object.type == "cuda":
        torch.cuda.set_device(device_object.index or 0)
    training_config.device = device
    torch.set_default_device(device)

    timing = None
    if mode == "benchmark":
        from compressarc.train.timing import TrainingTiming
        timing = TrainingTiming(device_object, args.benchmark_pass,
                                args.warmup_iterations, args.measured_iterations)

    def setup_call(metric_name, fn):
        return timing.setup_call(metric_name, fn) if timing is not None else fn()

    selection = [args.task] if args.task else range(10000)
    tasks = setup_call("setup/preprocessing_ms", lambda: preprocessing.preprocess_tasks(
        training_config.split, selection, multitensor_constraints=config.model.multitensor_constraints,
    ))
    if not tasks:
        raise ValueError(f"No tasks found for split={training_config.split!r}, task={args.task!r}")
    if args.save_checkpoints:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    for task in tasks:
        run = initialize_wandb(config, task) if timing is None else None
        with RunSummary(run, device_object):
            model = setup_call("setup/model_ms", lambda: ARCCompressor(task, config.model))
            optimizer = setup_call("setup/optimizer_ms", lambda: torch.optim.Adam(
                model.weights_list,
                lr=training_config.learning_rate,
                betas=tuple(training_config.optimizer_betas),
            ))
            tracker = SolutionTracker(task)
            if timing is not None:
                timing.finish_setup()
            if timing is not None:
                result = timing.train(task, model, optimizer, training_config, take_step)
                print(json.dumps({"task": task.task_name,
                                  "configuration": config.get("experiment", {}).get(
                                      "name", Path(args.config).stem if args.config else "default"),
                                  "pass": args.benchmark_pass, "metrics": result}, indent=2))
                return result

            with TrainingDiagnostics(run, model, task) as diagnostics, StepTimer(
                device_object, training_config.get("step_timing", "off"),
                args.output_dir / f"{task.task_name}.step_timing.jsonl",
            ) as step_timer:
                last_metrics = None
                progress = tqdm(range(training_config.iterations), desc=task.task_name, leave=False)
                for train_step in progress:
                    step_timer.start(train_step)
                    diagnostics.step = train_step
                    last_metrics = take_step(
                        task, model, optimizer, train_step, training_config,
                        return_outputs=True,
                        **({"phase_callback": step_timer.mark} if step_timer.mode == "phases" else {}),
                        **({"diagnostic_callback": diagnostics.observe} if run is not None else {}),
                    )
                    step_timer.mark("progress")
                    progress.set_postfix(loss=f"{last_metrics['loss']:.3f}")
                    step_timer.mark("diagnostics")
                    diagnostics.after_step()
                    step_timer.mark("candidates")
                    tracker.update(train_step, last_metrics["outputs"])
                    step_timer.mark("logging")
                    log_training_metrics(run, task, tracker, train_step, last_metrics)
                    step_timer.publish(step_timer.finish(), run)

                if args.save_checkpoints or run is not None:
                    args.output_dir.mkdir(parents=True, exist_ok=True)
                    checkpoint_path = args.output_dir / f"{task.task_name}.pt"
                    save_inference_checkpoint(
                        checkpoint_path, task, model, tracker, config,
                        training_config.iterations, last_metrics,
                    )
                    log_model_checkpoint(run, checkpoint_path)

                print(f"{task.task_name}: loss={last_metrics['loss']:.4f}")
                print("Guess 1:", tracker.solution_most_frequent)
                print("Guess 2:", tracker.solution_second_most_frequent)
                if task.solution_hash is not None:
                    print("Top-1 correct:", hash(tracker.solution_most_frequent) == task.solution_hash)
                    print("Pass@2 correct:", any(
                        hash(guess) == task.solution_hash
                        for guess in (tracker.solution_most_frequent, tracker.solution_second_most_frequent)
                    ))
                log_final_candidates(run, tracker)
            # Release the previous task before constructing the next GPU model.
            del model, optimizer, last_metrics



if __name__ == "__main__":
    main()




