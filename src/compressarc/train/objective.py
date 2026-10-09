"""ARC reconstruction objective and one optimizer update."""

import torch
import torch.nn.functional as functional
from config import load_config


def mask_select_logprobs(mask, length):
    """Return the partition and unnormalized scores for every valid slice."""
    logprobs = []
    for offset in range(mask.shape[0] - length + 1):
        logprob = -torch.sum(mask[:offset])
        logprob = logprob + torch.sum(mask[offset : offset + length])
        logprob = logprob - torch.sum(mask[offset + length :])
        logprobs.append(logprob)
    logprobs = torch.stack(logprobs, dim=0)
    return torch.logsumexp(logprobs, dim=0), logprobs


def take_step(task, model, optimizer, train_step, config=None, return_outputs=False,
              phase_callback=None):
    """Compute the ARC objective, update model weights, and return scalar metrics."""
    config = config if config is not None else load_config().training
    optimizer.zero_grad()
    if phase_callback is not None:
        phase_callback("forward")
    logits, x_mask, y_mask, kl_amounts, kl_names = model.forward()
    if phase_callback is not None:
        phase_callback("loss")
    logits = torch.cat([torch.zeros_like(logits[:, :1, :, :]), logits], dim=1)

    kl_components = {
        name: float(torch.sum(amount).detach().cpu())
        for amount, name in zip(kl_amounts, kl_names)
    }
    total_kl = sum(
        (torch.sum(kl_amount) for kl_amount in kl_amounts),
        start=logits.new_zeros(()),
    )
    reconstruction_error = logits.new_zeros(())

    for example_num in range(task.n_examples):
        for in_out_mode in range(2):
            if example_num >= task.n_train and in_out_mode == 1:
                continue

            grid_size_uncertain = not (
                task.in_out_same_size
                or task.all_out_same_size and in_out_mode == 1
                or task.all_in_same_size and in_out_mode == 0
            )
            if grid_size_uncertain:
                coefficient = config.shape_annealing_start ** max(
                    0, 1 - train_step / config.annealing_steps
                )
            else:
                coefficient = 1

            logits_slice = logits[example_num, :, :, :, in_out_mode]
            problem_slice = task.problem[example_num, :, :, in_out_mode]
            output_shape = task.shapes[example_num][in_out_mode]
            x_log_partition, x_logprobs = mask_select_logprobs(
                coefficient * x_mask[example_num, :, in_out_mode], output_shape[0]
            )
            y_log_partition, y_logprobs = mask_select_logprobs(
                coefficient * y_mask[example_num, :, in_out_mode], output_shape[1]
            )

            if grid_size_uncertain:
                x_log_partitions = [
                    mask_select_logprobs(
                        coefficient * x_mask[example_num, :, in_out_mode], length
                    )[0]
                    for length in range(1, x_mask.shape[1] + 1)
                ]
                y_log_partitions = [
                    mask_select_logprobs(
                        coefficient * y_mask[example_num, :, in_out_mode], length
                    )[0]
                    for length in range(1, y_mask.shape[1] + 1)
                ]
                x_log_partition = torch.logsumexp(torch.stack(x_log_partitions), dim=0)
                y_log_partition = torch.logsumexp(torch.stack(y_log_partitions), dim=0)

            logprobs_by_x = []
            for x_offset in range(x_logprobs.shape[0]):
                logprobs_by_y = []
                for y_offset in range(y_logprobs.shape[0]):
                    logprob = (
                        x_logprobs[x_offset] - x_log_partition
                        + y_logprobs[y_offset] - y_log_partition
                    )
                    logits_crop = logits_slice[
                        :,
                        x_offset : x_offset + output_shape[0],
                        y_offset : y_offset + output_shape[1],
                    ]
                    target_crop = problem_slice[: output_shape[0], : output_shape[1]]
                    logprob = logprob - functional.cross_entropy(
                        logits_crop.unsqueeze(0),
                        target_crop.unsqueeze(0),
                        reduction="sum",
                    )
                    logprobs_by_y.append(logprob)
                logprobs_by_x.append(torch.stack(logprobs_by_y))

            logprobs = torch.stack(logprobs_by_x)
            if grid_size_uncertain:
                coefficient = config.marginal_annealing_start ** max(
                    0, 1 - train_step / config.annealing_steps
                )
            else:
                coefficient = 1
            marginal_logprob = torch.logsumexp(
                coefficient * logprobs, dim=(0, 1)
            ) / coefficient
            reconstruction_error = reconstruction_error - marginal_logprob

    loss = config.kl_weight * total_kl + config.reconstruction_weight * reconstruction_error
    if phase_callback is not None:
        phase_callback("backward")
    loss.backward()
    if phase_callback is not None:
        phase_callback("optimizer")
    optimizer.step()

    if phase_callback is not None:
        phase_callback("metrics")
    metrics = {
        "kl": float(total_kl.detach().cpu()),
        "kl_components": kl_components,
        "reconstruction_error": float(reconstruction_error.detach().cpu()),
        "loss": float(loss.detach().cpu()),
    }
    if return_outputs:
        metrics["outputs"] = tuple(
            value.detach().cpu() for value in (logits, x_mask, y_mask)
        )
    if phase_callback is not None:
        phase_callback("end")
    return metrics
