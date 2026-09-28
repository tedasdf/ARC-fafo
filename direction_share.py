"""D4-equivariant direction mixing for CompressARC multitensors."""
import torch
import torch.nn as nn


class D4DirectionShare(nn.Module):
    """Most general scalar D4-equivariant map over eight directions."""

    def __init__(self):
        super().__init__()
        self.theta = nn.Parameter(torch.randn(10) * 0.01)
        keys = [
            (0, 0, 0), (0, 0, 2), (0, 0, 4),
            (1, 1, 0), (1, 1, 2), (1, 1, 4),
            (0, 1, 1), (0, 1, 3),
            (1, 0, 1), (1, 0, 3),
        ]
        key_to_id = {key: index for index, key in enumerate(keys)}
        orbit_id = torch.empty(8, 8, dtype=torch.long)
        for output_direction in range(8):
            for input_direction in range(8):
                delta = (input_direction - output_direction) % 8
                key = (output_direction % 2, input_direction % 2,
                       min(delta, 8 - delta))
                orbit_id[output_direction, input_direction] = key_to_id[key]
        self.register_buffer('orbit_id', orbit_id)

    def weight(self):
        return self.theta[self.orbit_id]

    def forward(self, x):
        return torch.einsum('oi,bi...->bo...', self.weight(), x)


def apply_d4(dims, x, model):
    """Apply a D4 module to CompressARC's direction axis without a residual."""
    direction_axis = sum(dims[:2])
    if direction_axis == 0:
        return model(x.unsqueeze(0)).squeeze(0)
    moved = x.movedim(direction_axis, 1)
    return model(moved).movedim(1, direction_axis)
