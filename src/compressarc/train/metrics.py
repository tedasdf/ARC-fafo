"""Training metric naming and prediction tracking."""

import ast

import numpy as np
import torch


def kl_metric_name(component_name):
    dims = ast.literal_eval(component_name)
    axis_names = ("example", "color", "direction", "height", "width")
    active_axes = [name for name, active in zip(axis_names, dims) if active]
    return "train/kl_components/" + "_".join(active_axes)


class SolutionTracker:
    """Maintain frequent test predictions and an EMA candidate, as in the old logger."""

    ema_decay = 0.97

    def __init__(self, task):
        self.task = task
        shape = (task.n_test, task.n_colors + 1, task.n_x, task.n_y)
        self.ema_logits = torch.zeros(shape)
        self.ema_x_mask = torch.zeros((task.n_test, task.n_x))
        self.ema_y_mask = torch.zeros((task.n_test, task.n_y))
        self.scores = {}
        self.solution_most_frequent = None
        self.solution_second_most_frequent = None

    def best_slice(self, mask, length):
        lengths = [length] if length is not None else range(1, mask.shape[0] + 1)
        best_score, best_start, best_end = None, 0, 0
        for candidate_length in lengths:
            scores = torch.stack([
                -torch.sum(mask[:offset])
                + torch.sum(mask[offset : offset + candidate_length])
                - torch.sum(mask[offset + candidate_length :])
                for offset in range(mask.shape[0] - candidate_length + 1)
            ])
            candidate_score = torch.max(scores)
            if best_score is None or candidate_score > best_score:
                best_score = candidate_score
                best_start = int(torch.argmax(scores))
                best_end = best_start + candidate_length
        return best_start, best_end

    def decode(self, logits, x_mask, y_mask):
        color_indices = torch.argmax(logits, dim=1)
        uncertainty = torch.logsumexp(logits, dim=1) - torch.amax(logits, dim=1)
        grids, uncertainties = [], []
        for index in range(self.task.n_test):
            fixed = self.task.in_out_same_size or self.task.all_out_same_size
            x_length = self.task.shapes[self.task.n_train + index][1][0] if fixed else None
            y_length = self.task.shapes[self.task.n_train + index][1][1] if fixed else None
            x0, x1 = self.best_slice(x_mask[index], x_length)
            y0, y1 = self.best_slice(y_mask[index], y_length)
            grid = color_indices[index, x0:x1, y0:y1].numpy()
            grids.append(tuple(tuple(self.task.colors[int(color)] for color in row) for row in grid))
            uncertainties.append(float(torch.mean(uncertainty[index, x0:x1, y0:y1])))
        solution = tuple(grids)
        return solution, float(np.mean(uncertainties))

    def update(self, train_step, outputs):
        logits, x_mask, y_mask = outputs
        logits = logits[self.task.n_train :, :, :, :, 1]
        x_mask = x_mask[self.task.n_train :, :, 1]
        y_mask = y_mask[self.task.n_train :, :, 1]
        self.ema_logits = self.ema_decay * self.ema_logits + (1 - self.ema_decay) * logits
        self.ema_x_mask = self.ema_decay * self.ema_x_mask + (1 - self.ema_decay) * x_mask
        self.ema_y_mask = self.ema_decay * self.ema_y_mask + (1 - self.ema_decay) * y_mask
        candidates = [
            (logits, x_mask, y_mask, 0),
            (self.ema_logits, self.ema_x_mask, self.ema_y_mask, -4),
        ]
        for cand_logits, cand_x, cand_y, penalty in candidates:
            solution, uncertainty = self.decode(cand_logits, cand_x, cand_y)
            score = -10 * uncertainty + penalty - (10 if train_step < 150 else 0)
            key = hash(solution)
            self.scores[key] = float(np.logaddexp(self.scores.get(key, -np.inf), score))
            self._update_top_two(key, solution)

    def _update_top_two(self, key, solution):
        if self.solution_most_frequent is None:
            self.solution_most_frequent = solution
        if self.solution_second_most_frequent is None:
            self.solution_second_most_frequent = solution
        first_key = hash(self.solution_most_frequent)
        second_key = hash(self.solution_second_most_frequent)
        if key != first_key and self.scores[key] >= self.scores.get(second_key, -np.inf):
            self.solution_second_most_frequent = solution
            if self.scores[key] >= self.scores.get(first_key, -np.inf):
                self.solution_second_most_frequent = self.solution_most_frequent
                self.solution_most_frequent = solution


def training_metrics_payload(train_step, metrics):
    """Flatten one optimizer-step result into W&B scalar metrics."""
    payload = {
        "train_step": train_step,
        "train/loss": metrics["loss"],
        "train/reconstruction_error": metrics["reconstruction_error"],
        "train/total_KL": metrics["kl"],
    }
    payload.update({
        kl_metric_name(name): value
        for name, value in metrics["kl_components"].items()
    })
    return payload
