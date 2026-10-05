"""Train ARCCompressor tasks locally or on Modal and optionally log to W&B."""

import argparse
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

import preprocessing
from config import load_config
from compressarc.model.model import ARCCompressor
from compressarc.train.logging import (
    initialize_wandb,
    log_final_results,
    log_latent_pca,
    log_model_checkpoint,
    log_problem,
    log_training_step,
)
from compressarc.train import take_step
from compressarc.train.metrics import SolutionTracker


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="See TRAINING.md for Modal setup and examples.",
    )
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
                        help="Save final .pt checkpoints and upload model artifacts when W&B is enabled")
    return parser.parse_args()


def merge_cli_config(args):
    return load_config(args.config, args.overrides)


def main():
    args = parse_args()
    config = merge_cli_config(args)
    training_config, logging_config = config.training, config.logging
    if training_config.iterations < 1 or training_config.learning_rate <= 0 or training_config.annealing_steps < 1:
        raise ValueError("iterations, learning_rate, and annealing_steps must be positive")
    if logging_config.latent_pca and (
        logging_config.pca_samples < 1
        or logging_config.pca_components < 1
        or logging_config.pca_max_panels < 1
    ):
        raise ValueError("PCA sample, component, and panel counts must be positive")
    if logging_config.wandb_log_every < 1 or logging_config.prediction_every < 1:
        raise ValueError("W&B and prediction logging intervals must be positive")

    if args.backend == "modal":
        from compressarc.train.modal_launcher import launch_modal

        launch_modal(config, args)
        return
    np.random.seed(training_config.seed)
    torch.manual_seed(training_config.seed)
    device = training_config.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but no CUDA device is available")
    training_config.device = device
    torch.set_default_device(device)

    selection = [args.task] if args.task else range(10000)
    tasks = preprocessing.preprocess_tasks(training_config.split, selection, multitensor_constraints=config.model.multitensor_constraints)
    if not tasks:
        raise ValueError(f"No tasks found for split={training_config.split!r}, task={args.task!r}")
    if args.save_checkpoints:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    for task in tasks:
        model = ARCCompressor(task, config.model)
        optimizer = torch.optim.Adam(
            model.weights_list,
            lr=training_config.learning_rate,
            betas=tuple(training_config.optimizer_betas),
        )
        tracker = SolutionTracker(task)
        run = initialize_wandb(config, task, model, optimizer)
        log_problem(run, task)

        last_metrics = None
        try:
            progress = tqdm(range(training_config.iterations), desc=task.task_name, leave=False)
            for train_step in progress:
                prediction_step = (
                    (train_step + 1) % logging_config.prediction_every == 0
                    or train_step == training_config.iterations - 1
                )
                last_metrics = take_step(
                    task, model, optimizer, train_step, training_config,
                    return_outputs=prediction_step,
                )
                progress.set_postfix(loss=f"{last_metrics['loss']:.3f}")
                if prediction_step:
                    tracker.update(train_step, last_metrics["outputs"])

                if run is not None and (
                    train_step % logging_config.wandb_log_every == 0 or prediction_step
                ):
                    log_training_step(
                        run, task, tracker, train_step, last_metrics,
                        include_prediction=prediction_step,
                    )

            if args.save_checkpoints:
                checkpoint_path = args.output_dir / f"{task.task_name}.pt"
                serial_metrics = {key: value for key, value in last_metrics.items() if key not in ("outputs", "kl_components")}
                torch.save(
                    {
                        "task": task.task_name,
                        "split": training_config.split,
                        "config": OmegaConf.to_container(config, resolve=True),
                        "weights": [weight.detach().cpu() for weight in model.weights_list],
                        "metrics": serial_metrics,
                    },
                    checkpoint_path,
                )
                log_model_checkpoint(
                    run, checkpoint_path,
                    metadata={
                        "task_name": task.task_name,
                        "split": training_config.split,
                        "iterations_completed": training_config.iterations,
                        "model": OmegaConf.to_container(config.model, resolve=True),
                    },
                )

            if run is not None:
                log_final_results(run, task, tracker)
                log_latent_pca(run, model, logging_config)

            print(f"{task.task_name}: loss={last_metrics['loss']:.4f}")
            print("Guess 1:", tracker.solution_most_frequent)
            print("Guess 2:", tracker.solution_second_most_frequent)
            if task.solution_hash is not None:
                print("Top-1 correct:", hash(tracker.solution_most_frequent) == task.solution_hash)
                print("Pass@2 correct:", any(
                    hash(guess) == task.solution_hash
                    for guess in (tracker.solution_most_frequent, tracker.solution_second_most_frequent)
                ))
        finally:
            if run is not None:
                run.finish()



if __name__ == "__main__":
    main()




