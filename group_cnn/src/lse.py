import torch
import torch.nn as nn

import numpy as np 

from static_op import diagonal_shift_

def cummax_(x, dim, masks):
    masks = 1e3*(1-masks)
    max_ = torch.max(x-masks, dim=dim, keepdim=True)[0] + masks + 1e-3
    min_ = torch.min(x+masks, dim=dim, keepdim=True)[0] - masks - 1e-3
    print(max_, min_)
    x = torch.cummax(x-masks, dim=dim)[0] + masks
    return (x - min_) / (max_-min_) * 2 - 1

def diagonal_cummax_(x, dim1, dim2, masks):
    masks_ = 1e3*(1-masks)
    min_dim = min(x.shape[dim1], x.shape[dim2])
    n_iters = int(np.ceil(np.log2(min_dim)))
    # compute the cummax and max via forward+backward associative scan
    max_x = x - masks_
    for sign in (1, -1):
        for i in range(n_iters):
            shift_amount = sign*2**i
            shifted_x = diagonal_shift_(max_x, dim1, dim2, masks_, shift_amount=shift_amount, pad_value=-1e3)
            max_x = torch.max(max_x, shifted_x)
        if sign == 1:  # save the cummax after the forward associative scan
            cummax_x = max_x + masks_
    max_x = max_x + masks_
    # compute the min via forward+backward associative scan
    min_x = x + masks_
    for sign in (1, -1):
        for i in range(n_iters):
            shift_amount = sign*2**i
            shifted_x = diagonal_shift_(min_x, dim1, dim2, masks_, shift_amount=shift_amount, pad_value=1e3)
            min_x = torch.min(min_x, shifted_x)
    min_x = min_x - masks_
    return ((cummax_x - min_x) / (max_x-min_x+1e-5) * 2 - 1)*masks  # rescale the cummax to fit the max and min


class MorphologicalMax(nn.Module):
    def __init__(self, H, W, tau=0.1):
        super().__init__()
        L_min = min(H, W)
        L_max = max(H, W)

        self.tau = tau
        self.kernel = nn.Parameter(torch.zeros(L_max))
        self.diagonal_kernel = nn.Parameter(torch.zeros(L_min))
        
        support = torch.ones(L_max, dtype=torch.bool)
        self.register_buffer("support", support)

    def diagonal_phi(self, x, mark=None):
        """
        Canonical diagonal morphological cummax.

        x: [B, H, W]

        Returns:
            out:   [B, H, W]  LSE version
            exact: [B, H, W]  exact max version
        """
        B, H, W = x.shape
        if mark:
            mark('start')

        D = H + W - 1
        L = min(H, W)

        # --------------------------------------------------
        # 1. Pack NW-SE diagonals into rows
        #    [B, D, L]
        # --------------------------------------------------

        diagonals = torch.full(
            (B, D, L),
            float("-inf"),
            device=x.device,
            dtype=x.dtype,
        )
        if mark:
            mark('allocate_diagonals')

        # remember where every packed value came from
        coordinates = []

        for d in range(-(H - 1), W):

            coords = []

            for h in range(H):
                w = h + d

                if 0 <= w < W:
                    coords.append((h, w))

            coordinates.append(coords)
        if mark:
            mark('build_coordinates')

        for d, coords in enumerate(coordinates):
            for p, (h, w) in enumerate(coords):
                diagonals[:, d, p] = x[:, h, w]
        if mark:
            mark('pack_diagonals')

        # --------------------------------------------------
        # 2. Build f(x-u)
        #    [B, D, output_position, offset]
        # --------------------------------------------------

        shifted = torch.full(
            (B, D, L, L),
            float("-inf"),
            device=x.device,
            dtype=x.dtype,
        )
        if mark:
            mark('allocate_shifted')

        for d, coords in enumerate(coordinates):
            diag_len = len(coords)

            for i in range(diag_len):
                for u in range(i + 1):
                    shifted[:, d, i, u] = diagonals[:, d, i - u]
        if mark:
            mark('build_shifted')

        # --------------------------------------------------
        # 3. Morphological kernel
        # --------------------------------------------------

        scores = shifted + self.diagonal_kernel.view(1, 1, 1, L)
        if mark:
            mark('add_kernel')

        # Exact morphological max
        exact_diag = scores.max(dim=-1).values
        if mark:
            mark('exact_max')

        # Smooth max
        out_diag = self.tau * torch.logsumexp(
            scores / self.tau,
            dim=-1,
        )
        if mark:
            mark('smooth_logsumexp')

        # --------------------------------------------------
        # 4. Scatter diagonals back into [B, H, W]
        # --------------------------------------------------

        out = torch.empty_like(x)
        exact = torch.empty_like(x)

        for d, coords in enumerate(coordinates):
            for p, (h, w) in enumerate(coords):

                out[:, h, w] = out_diag[:, d, p]
                exact[:, h, w] = exact_diag[:, d, p]
        if mark:
            mark('allocate_and_scatter')

        return out, exact



    def phi(self, x, mark=None):
        if mark:
            mark('start')
        # canonical direction is always last dim
        original_shape = x.shape
        W = x.shape[-1]

        x_flat = x.reshape(-1, W)
        if mark:
            mark('reshape_input')

        x_shifted = torch.full(
            (x_flat.shape[0], W, W),
            float("-inf"),
            device=x.device,
            dtype=x.dtype
        )
        if mark:
            mark('allocate_shifted')

        for i in range(W):
            for u in range(i + 1):
                x_shifted[:, i, u] = x_flat[:, i - u]
        if mark:
            mark('build_shifted')

        kernel = self.kernel[:W]
        support = self.support[:W]
        scores = x_shifted + kernel.view(1, 1, W)
        if mark:
            mark('add_kernel')

        scores = scores.masked_fill(
            ~support.view(1, 1, W),
            float("-inf")
        )
        if mark:
            mark('apply_support')

        exact = scores.max(dim=-1).values
        if mark:
            mark('exact_max')

        out = self.tau * torch.logsumexp(
            scores / self.tau,
            dim=-1
        )
        if mark:
            mark('smooth_logsumexp')

        out = out.reshape(original_shape)
        exact = exact.reshape(original_shape)
        if mark:
            mark('reshape_output')

        return out, exact

    def forward(self, x):
        # x: [B, H, W]

        # ==================================================
        # Cardinal directions
        # ==================================================
        start = torch.cuda.Event(enable_timing=True) 
        end = torch.cuda.Event(enable_timing=True)


        start.record()
        right, exact_right = self.phi(x)
        end.record()

        x_rot = torch.rot90(x, -1, dims=(1, 2))
        up, _ = self.phi(x_rot)
        up = torch.rot90(up, 1, dims=(1, 2))

        x_rot = torch.rot90(x, 2, dims=(1, 2))
        left, _ = self.phi(x_rot)
        left = torch.rot90(left, 2, dims=(1, 2))

        x_rot = torch.rot90(x, 1, dims=(1, 2))
        down, _ = self.phi(x_rot)
        down = torch.rot90(down, -1, dims=(1, 2))

        torch.cuda.synchronize()
      

        # ==================================================
        # Diagonal directions
        # ==================================================
        time_start = torch.cuda.Event(enable_timing=True) 
        time_end = torch.cuda.Event(enable_timing=True)

        time_start.record()
        top_right, exact_top_right = self.diagonal_phi(x)
        time_end.record()
        
        x_rot = torch.rot90(x, -1, dims=(1, 2))
        top_left, _ = self.diagonal_phi(x_rot)
        top_left = torch.rot90(
            top_left,
            1,
            dims=(1, 2)
        )

        x_rot = torch.rot90(x, 2, dims=(1, 2))
        bottom_left, _ = self.diagonal_phi(x_rot)
        bottom_left = torch.rot90(
            bottom_left,
            2,
            dims=(1, 2)
        )

        x_rot = torch.rot90(x, 1, dims=(1, 2))
        bottom_right, _ = self.diagonal_phi(x_rot)
        bottom_right = torch.rot90(
            bottom_right,
            -1,
            dims=(1, 2)
        )
       

        torch.cuda.synchronize()

        print(f"phi: {start.elapsed_time(end):.3f} ms")
        print(f"diagonal_phi: {time_start.elapsed_time(time_end):.3f} ms")
        # ==================================================
        # Stack all 8 directions
        # ==================================================

        out = torch.stack(
            [
                right,
                top_right,
                up,
                top_left,
                left,
                bottom_left,
                down,
                bottom_right,
            ],
            dim=1
        )

        return out, exact_right, exact_top_right

def benchmark(fn, x, warmup=20, runs=100):
    # warm up
    for _ in range(warmup):
        fn(x)

    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()

    for _ in range(runs):
        fn(x)

    end.record()

    torch.cuda.synchronize()

    return start.elapsed_time(end) / runs

if __name__ == "__main__":

    x = torch.tensor([[
        [1., 4., 2., 7., 3.],
        [5., 1., 8., 2., 6.],
        [0., 9., 3., 4., 2.],
        [6., 2., 1., 5., 4.],
        [3., 7., 2., 1., 8.],
    ]])

    model = MorphologicalMax(
        H=x.shape[-2],
        W=x.shape[-1],
        tau=0.1
    )
    x = x.cuda()
    model = model.cuda()
    # actual, exact_right, exact_top_right = model(x)

    # right = torch.cummax(x, dim=-1).values

    # print("input:\n", x)
    # print("output shape:", actual.shape)

    # print("\nexpected right:\n", right)
    # print("\nexact right:\n", exact_right)

    # print("\nexact diagonal:\n", exact_top_right)

    # assert torch.allclose(exact_right, right)

    # assert actual.shape == (
    #     x.shape[0],
    #     8,
    #     x.shape[1],
    #     x.shape[2],
    # )
    phi_ms = benchmark(model.phi, x)
    diag_ms = benchmark(model.diagonal_phi, x)

    print("phi:", phi_ms, "ms")
    print("diagonal:", diag_ms, "ms")


    diag_ms = benchmark(model.diagonal_phi, x)
    phi_ms = benchmark(model.phi, x)

    print("phi:", phi_ms, "ms")
    print("diagonal:", diag_ms, "ms")

