"""Vendored from ARC-fafo/group_cnn/src/group_cnn.py and shift_adapter.py."""
import torch
import torch.nn as nn
import torch.nn.functional as F

class TiedDirectionalConv(nn.Module):
    def __init__(self):
        super().__init__()

        # canonical cardinal kernel
        self.cardinal_kernel = nn.Parameter(torch.tensor([
            [0., 0., 0.],
            [1., 0., 0.],
            [0., 0., 0.]
        ]))

        # canonical diagonal kernel
        self.diagonal_kernel = nn.Parameter(torch.tensor([
            [1., 0., 0.],
            [0., 0., 0.],
            [0., 0., 0.]
        ]))

    def forward(self, x):
        """
        x: [B, 2, 4, H, W]

        x[:, 0] = cardinal directions
        x[:, 1] = diagonal directions
        """

        # --------------------------------
        # Cardinal orbit
        # --------------------------------
        right = self.cardinal_kernel
        up = torch.rot90(
            self.cardinal_kernel,
            1,
            dims=(0, 1)
        )
        left = torch.rot90(
            self.cardinal_kernel,
            2,
            dims=(0, 1)
        )
        down = torch.rot90(
            self.cardinal_kernel,
            3,
            dims=(0, 1)
        )

        cardinal = torch.stack([
            up,
            left,
            down,
            right
        ])                          # [4, 3, 3]

        # --------------------------------
        # Diagonal orbit
        # --------------------------------
        diag0 = self.diagonal_kernel
        diag1 = torch.rot90(
            self.diagonal_kernel,
            1,
            dims=(0, 1)
        )
        diag2 = torch.rot90(
            self.diagonal_kernel,
            2,
            dims=(0, 1)
        )
        diag3 = torch.rot90(
            self.diagonal_kernel,
            3,
            dims=(0, 1)
        )

        diagonal = torch.stack([
            diag0,
            diag1,
            diag2,
            diag3,
        ])                          # [4, 3, 3]

        # --------------------------------
        # Two D4 orbits
        # --------------------------------
        kernels = torch.stack([
            cardinal,
            diagonal
        ])                          # [2, 4, 3, 3]

        B, _, _, H, W = x.shape

        # conv2d wants channels rather than [orbit, direction]
        x_flat = x.reshape(B, 8, H, W)

        # [2,4,3,3] -> [8,1,3,3]
        kernels_flat = kernels.reshape(8, 1, 3, 3)

        y = F.conv2d(
            x_flat,
            kernels_flat,
            padding=1,
            groups=8,
        )

        # restore [cardinal/diagonal, direction]
        y = y.reshape(B, 2, 4, H, W)

        return y




def tied_directional_shift(x, masks, model):
    """Map [E, (C), D, H, W, F] to TiedDirectionalConv and back.

    Model cardinal order: up, left, down, right.
    Model diagonal order: down-right, up-right, up-left, down-left.
    Adapted to this checkout's even/odd feature selection in both orbits.
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
            # This checkout selects channel_split::2 for both orbit types.
            feature_start = half
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
