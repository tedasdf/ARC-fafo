"""Puzzle, prediction, W&B, and optional latent PCA logging helpers."""

import ast

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from compressarc.model.multitensor_systems import multify
from compressarc.analysis.visualization import plot_pca_component, plot_predictions, plot_problem
from .metrics import training_metrics_payload


def initialize_wandb(config, task, model, optimizer):
    if not config.logging.wandb:
        return None
    try:
        import wandb
    except ImportError as error:
        raise RuntimeError("W&B is enabled but wandb is not installed; install the project requirements.") from error
    run = wandb.init(
        project=config.logging.wandb_project,
        entity=config.logging.wandb_entity,
        name=f"{task.task_name}-{config.training.split}",
        mode=config.logging.wandb_mode,
        config={
            "task_name": task.task_name,
            "split": config.training.split,
            "n_train_examples": task.n_train,
            "n_test_examples": task.n_test,
            "n_examples": task.n_examples,
            "n_colors": task.n_colors + 1,
            "canvas_height": task.n_x,
            "canvas_width": task.n_y,
            "model": OmegaConf.to_container(config.model, resolve=True),
            "training": OmegaConf.to_container(config.training, resolve=True),
            "optimizer": type(optimizer).__name__,
            "parameter_count": sum(p.numel() for p in model.weights_list),
        },
        tags=["arc-agi", config.training.split, *list(config.logging.wandb_tags)],
        save_code=True,
    )
    run.define_metric("train_step")
    run.define_metric("train/*", step_metric="train_step")
    run.define_metric("predictions/*", step_metric="train_step")
    return run


def log_latent_pca(run, model, logging_config):
    import matplotlib.pyplot as plt
    if run is None or not logging_config.latent_pca:
        return
    import wandb

    @multify
    def to_cpu(dims, value):
        return value.detach().cpu()

    @multify
    def add_tensors(dims, left, right):
        return left + right

    @multify
    def center(dims, total):
        mean = total / logging_config.pca_samples
        return mean - mean.mean(dim=tuple(range(mean.ndim - 1)))

    total = None
    last_kl, names = None, None
    with torch.no_grad():
        for _ in tqdm(range(logging_config.pca_samples), desc="latent PCA", leave=False):
            sample, last_kl, names = model.latent_decoder(
                model.target_capacities, model.decode_weights, model.multiposteriors
            )
            sample = to_cpu(sample)
            total = sample if total is None else add_tensors(total, sample)
    means = center(total)
    axes_names = ("example", "color", "direction", "height", "width")
    table = wandb.Table(columns=["tensor_dims", "semantic_axes", "tensor_shape", "KL", "component", "strength", "heatmap"])
    significant_tensor_count = 0
    for kl, name in zip(last_kl, names):
        kl_value = float(torch.sum(kl))
        if kl_value < logging_config.pca_kl_threshold:
            continue
        significant_tensor_count += 1
        dims = tuple(ast.literal_eval(name))
        tensor = means[dims].numpy()
        flat = tensor.reshape(-1, tensor.shape[-1])
        vectors, singular_values, _ = np.linalg.svd(flat, full_matrices=False)
        active_axes = [axis for axis, active in zip(axes_names, dims) if active]
        for component in range(min(logging_config.pca_components, len(singular_values))):
            panel = vectors[:, component].reshape(tensor.shape[:-1])
            strength = float(singular_values[component] / flat.shape[0])
            figure = plot_pca_component(
                panel, active_axes, component + 1, strength, logging_config.pca_max_panels
            )
            table.add_data(
                str(dims), ", ".join(active_axes), str(tensor.shape), kl_value,
                component + 1, strength, wandb.Image(figure)
            )
            plt.close(figure)
    run.log({"latent_analysis/principal_components": table})
    run.summary["pca_significant_tensor_count"] = significant_tensor_count
    run.summary["pca_kl_threshold"] = logging_config.pca_kl_threshold




def log_problem(run, task):
    """Log the ARC puzzle grid montage once at the beginning of a run."""
    if run is None:
        return
    import matplotlib.pyplot as plt
    import wandb

    figure = plot_problem(task, fname=False)
    run.log({"puzzle/problem": wandb.Image(figure)})
    plt.close(figure)


def log_training_step(run, task, tracker, train_step, metrics, include_prediction=False):
    """Log scalar training metrics and, on selected steps, prediction images."""
    if run is None:
        return
    import matplotlib.pyplot as plt
    import wandb

    payload = training_metrics_payload(train_step, metrics)
    if include_prediction:
        figure = plot_predictions(
            task, tracker.solution_most_frequent, tracker.solution_second_most_frequent
        )
        payload["predictions/solutions"] = wandb.Image(figure)
        plt.close(figure)
    run.log(payload)


def log_final_results(run, task, tracker):
    """Add the final two guesses and available accuracy checks to the W&B summary."""
    if run is None:
        return
    run.summary["guess_1"] = [
        [[int(color) for color in row] for row in grid]
        for grid in tracker.solution_most_frequent
    ]
    run.summary["guess_2"] = [
        [[int(color) for color in row] for row in grid]
        for grid in tracker.solution_second_most_frequent
    ]
    if task.solution_hash is not None:
        guess_1_correct = hash(tracker.solution_most_frequent) == task.solution_hash
        guess_2_correct = hash(tracker.solution_second_most_frequent) == task.solution_hash
        run.summary["top_1_correct"] = guess_1_correct
        run.summary["pass_2_correct"] = guess_1_correct or guess_2_correct
