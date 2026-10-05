"""Multitensor adapter and projected residual tied-convolution layer."""
import torch
from torch import nn

from ..helper import apply_residual, only_do_for_certain_shapes
from .morphological import TiedDirectionalConv


def tied_directional_shift(x, masks, model):
    """Map [E, (C), D, H, W, F] to TiedDirectionalConv and back.

    Model cardinal order: up, left, down, right.
    Model diagonal order: down-right, up-right, up-left, down-left.
    This reproduces the original wrapper's even-feature cardinal selection.
    """
    if x.ndim not in (5, 6) or x.shape[-4] != 8 or x.shape[-1] % 2:
        raise ValueError("Expected [E, (C), 8, H, W, even_features]")
    height, width = x.shape[-3:-1]
    valid = 1 - (1 - masks[..., 0]) * (1 - masks[..., 1])
    valid = valid.reshape(x.shape[0], *([1] * (x.ndim - 4)), height, width, 1)
    masked = x * valid
    # For each original direction slot, its index in the flattened CNN orbit.
    movement_indices = (
        (2, 4, 3, 5, 0, 6, 1, 7),  # first feature half
        (0, 6, 1, 7, 2, 4, 3, 5),  # opposite movements
    )
    halves = []
    for half, mapping in enumerate(movement_indices):
        by_movement = [None] * 8
        for direction, movement in enumerate(mapping):
            # Cardinal directions use even features in BOTH output halves.
            feature_start = 0 if direction % 2 == 0 else half
            grid = masked[..., direction, :, :, feature_start::2]
            by_movement[movement] = grid.movedim(-1, -3)
        # [E, (C), F/2, 8, H, W] -> [E*(C)*F/2, 2, 4, H, W]
        packed = torch.stack(by_movement, dim=-3)
        shifted = model(packed.reshape(-1, 2, 4, height, width))
        unpacked = shifted.reshape(packed.shape)
        restored = [unpacked[..., movement, :, :].movedim(-3, -1)
                    for movement in mapping]
        halves.append(torch.stack(restored, dim=-4))
    return torch.cat(halves, dim=-1)



class TiedShiftLayer(nn.Module):
    """Use one trainable tied convolution per model depth."""

    def __init__(self, multify, n_layers=1):
        super().__init__()
        self.models = nn.ModuleList([TiedDirectionalConv() for _ in range(n_layers)])

        def apply_one(dims, x, weights, masks, *, layer_index=0, **kwargs):
            return apply_residual(
                x, weights,
                lambda z: tied_directional_shift(z, masks, self.models[layer_index]),
                **kwargs,
            )

        self._apply_multitensor = multify(
            only_do_for_certain_shapes((1, 1, 1, 1, 1), (1, 0, 1, 1, 1))(apply_one)
        )

    def forward(self, x, weights, masks, *, layer_index=0, **kwargs):
        return self._apply_multitensor(x, weights, masks, layer_index=layer_index, **kwargs)
