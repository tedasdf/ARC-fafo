"""Puzzle, prediction, W&B, and optional latent PCA logging helpers."""

import ast
import json
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from compressarc.model.multitensor_systems import multify
from compressarc.analysis.visualization import color_list, plot_pca_component, plot_predictions, plot_problem
from .metrics import training_metrics_payload


def initialize_wandb(config, task, model, optimizer):
    if not config.logging.wandb:
        return None
    try:
        import wandb
    except ImportError as error:
        raise RuntimeError("W&B is enabled but wandb is not installed; install the project requirements.") from error
    configuration = config.get("experiment", {}).get("name")
    run = wandb.init(
        project=config.logging.wandb_project,
        entity=config.logging.wandb_entity,
        name=f"{task.task_name}-{config.training.split}" + (f"-{configuration}" if configuration else ""),
        mode=config.logging.wandb_mode,
        config={
            "task_name": task.task_name,
            "configuration": configuration,
            "split": config.training.split,
            "n_train_examples": task.n_train,
            "n_test_examples": task.n_test,
            "n_examples": task.n_examples,
            "n_colors": task.n_colors + 1,
            "canvas_height": task.n_x,
            "canvas_width": task.n_y,
            "model": OmegaConf.to_container(config.model, resolve=True),
            "training": OmegaConf.to_container(config.training, resolve=True),
            "logging": OmegaConf.to_container(config.logging, resolve=True),
            "optimizer": type(optimizer).__name__,
            "parameter_count": sum(p.numel() for p in model.weights_list),
        },
        tags=["arc-agi", config.training.split, *list(config.logging.wandb_tags)],
        save_code=True,
    )
    run.define_metric("train_step")
    run.define_metric("train/*", step_metric="train_step")
    run.define_metric("predictions/*", step_metric="train_step")
    run.define_metric("ranking/*", step_metric="train_step")
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




def grid_image(grid):
    """RGB pixels in original ARC colors, without color-index remapping."""
    return color_list[np.asarray(grid, dtype=np.int64)].astype(np.uint8)


def reference_outputs_table(task):
    """Known training outputs and available full test references, logged once."""
    import wandb
    table = wandb.Table(columns=["split", "example", "height", "width", "output", "grid"])
    train_grids = [case["output"] for case in task.unprocessed_problem["train"]]
    for split, grids in (("train", train_grids), ("test", task.reference_solutions or ())):
        for index, grid in enumerate(grids):
            raw = np.asarray(grid, dtype=np.int64)
            table.add_data(split, index + 1, raw.shape[0], raw.shape[1],
                           wandb.Image(grid_image(raw)), raw.tolist())
    return table


def training_prediction_records(task, outputs, tracker):
    """Original-color y_true/y_pred and their output-mode crop for train cases."""
    logits, x_mask, y_mask = (value.detach().cpu() for value in outputs)
    colors = np.asarray(task.colors)
    result = []
    for index, case in enumerate(task.unprocessed_problem["train"]):
        expected = np.asarray(case["output"], dtype=np.int64)
        height, width = expected.shape
        x0, x1 = tracker.best_slice(x_mask[index, :, 1], height)
        y0, y1 = tracker.best_slice(y_mask[index, :, 1], width)
        indices = torch.argmax(logits[index, :, x0:x1, y0:y1, 1], dim=0).numpy()
        predicted = colors[indices]
        result.append({"example": index + 1, "y_true": expected.tolist(), "y_pred": predicted.tolist(),
                       "exact_match": bool(np.array_equal(expected, predicted)),
                       "crop": {"row_start": x0, "row_end": x1, "column_start": y0, "column_end": y1}})
    return result


def training_predictions_table(task, outputs, tracker, records=None):
    import wandb
    table = wandb.Table(columns=["example", "expected_y", "predicted_y", "exact_match",
                                 "expected_grid", "predicted_grid"])
    for record in records if records is not None else training_prediction_records(task, outputs, tracker):
        table.add_data(record["example"], wandb.Image(grid_image(record["y_true"])),
                       wandb.Image(grid_image(record["y_pred"])), record["exact_match"],
                       record["y_true"], record["y_pred"])
    return table


def log_problem(run, task):
    """Log the ARC puzzle grid montage once at the beginning of a run."""
    if run is None:
        return
    import matplotlib.pyplot as plt
    import wandb

    figure = plot_problem(task, fname=False)
    run.log({"puzzle/problem": wandb.Image(figure),
             "puzzle/reference_outputs": reference_outputs_table(task)})
    plt.close(figure)


def ranking_tables(diagnostics):
    """Events reference IDs; newly seen candidate grids appear once in a registry."""
    import wandb
    candidate_fields = ("contribution_index", "source", "id", "uncertainty", "confidence_score",
        "early_step_penalty", "ema_penalty", "contribution_score", "accumulated_before", "accumulated_after",
        "top1_after", "top1_after_score", "top2_after", "top2_after_score", "top1_changed", "top2_changed")
    selected_fields = ("rank", "id", "accumulated_score", "raw_count", "ema_count", "raw_log_mass",
        "ema_log_mass", "last_source", "last_uncertainty", "last_contribution", "last_top1_promotion_source")
    candidates = wandb.Table(columns=["train_step", "iteration", *candidate_fields])
    selected = wandb.Table(columns=["train_step", "iteration", *selected_fields])
    grids = wandb.Table(columns=["id", "first_train_step", "grids_json"])
    for record in diagnostics.pending:
        for candidate in record["candidates"]:
            candidates.add_data(record["train_step"], record["iteration"], *[candidate[key] for key in candidate_fields])
        for selection in record["selected"]:
            selected.add_data(record["train_step"], record["iteration"], *[selection[key] for key in selected_fields])
    for entry in diagnostics.pending_grids:
        grids.add_data(entry["id"], entry["first_train_step"], json.dumps(entry["grid"]))
    return candidates, selected, grids


def log_ranking_trace(run, directory):
    import wandb
    artifact = wandb.Artifact(name=f"ranking-{run.id}", type="ranking-diagnostics")
    for filename in ("events.jsonl", "grids.jsonl", "metadata.json", "training_predictions.jsonl"):
        artifact.add_file(str(directory / filename), name=filename)
    run.log_artifact(artifact)


def log_training_step(run, task, tracker, train_step, metrics, include_prediction=False,
                      include_accuracy=False):
    """Log correctness separately from the optional prediction-image cadence."""
    if run is None:
        return
    import matplotlib.pyplot as plt
    import wandb

    payload = training_metrics_payload(train_step, metrics)
    diagnostics = getattr(tracker, "diagnostics", None)
    if diagnostics is not None and diagnostics.last is not None:
        payload.update(diagnostics.scalar_metrics())
    if (include_accuracy or include_prediction) and task.solution_hash is not None:
        first_correct = hash(tracker.solution_most_frequent) == task.solution_hash
        second_correct = hash(tracker.solution_second_most_frequent) == task.solution_hash
        payload["predictions/top_1_correct"] = int(first_correct)
        payload["predictions/pass_2_correct"] = int(first_correct or second_correct)
    if include_prediction:
        figure = plot_predictions(
            task, tracker.solution_most_frequent, tracker.solution_second_most_frequent
        )
        payload["predictions/solutions"] = wandb.Image(figure)
        payload["train/predictions"] = training_predictions_table(task, metrics["outputs"], tracker, metrics.get("training_prediction_records"))
        if diagnostics is not None:
            payload["ranking/candidates"], payload["ranking/selected"], payload["ranking/grids"] = ranking_tables(diagnostics)
            candidate_figure = plot_predictions(task, *[diagnostics.grids[entry["id"]] for entry in diagnostics.last["candidates"]])
            payload["ranking/raw_vs_ema"] = wandb.Image(candidate_figure, caption="Raw (left), EMA (right)")
            plt.close(candidate_figure)
        plt.close(figure)
    run.log(payload)
    if include_prediction and diagnostics is not None:
        diagnostics.pending.clear()
        diagnostics.pending_grids.clear()


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



def log_model_checkpoint(run, checkpoint_path, *, metadata=None):
    """Attach a saved .pt checkpoint to an active W&B run as a model artifact."""
    if run is None:
        return None
    import wandb

    checkpoint_path = Path(checkpoint_path).resolve(strict=True)
    artifact = wandb.Artifact(
        name=f"model-{run.id}",
        type="model",
        metadata={} if metadata is None else dict(metadata),
    )
    artifact.add_file(str(checkpoint_path), name=checkpoint_path.name)
    return run.log_artifact(artifact, aliases=["latest"])
