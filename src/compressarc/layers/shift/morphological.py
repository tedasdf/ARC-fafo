
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



