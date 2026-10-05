"""Integration checks for the primitive shift layer's assembly."""

from itertools import product

import pytest
import torch

from compressarc.layers.shift.primitives import ShiftPrimitives
from compressarc.model.multitensor_systems import MultiTensorSystem, multify


ACTIVE_SHAPES = {(1, 1, 1, 1, 1), (1, 0, 1, 1, 1)}


def reference_normalize(x):
    axes = tuple(range(x.ndim - 1))
    centered = x - x.mean(dim=axes)
    return centered / torch.sqrt(1e-8 + centered.square().mean(dim=axes))


def reference_directions(x, masks, has_colors):
    """Place values at explicit destinations without using the layer helpers."""
    valid = masks.any(dim=-1)
    if has_colors:
        valid = valid[:, None]
    masked = x * valid.unsqueeze(-3).unsqueeze(-1)
    result = torch.zeros_like(x)
    # Direction slots represent down, down-right, right, up-right, then reverses.
    deltas = [(1, 0), (1, 1), (0, 1), (-1, 1),
              (-1, 0), (-1, -1), (0, -1), (1, -1)]
    for direction, (dr, dc) in enumerate(deltas):
        for half in range(2):
            row_delta, col_delta = (dr, dc) if half == 0 else (-dr, -dc)
            # Cardinal slots select even channels; diagonal slots select each half.
            source_half = 0 if direction % 2 == 0 else half
            source = masked[..., direction, :, :, source_half::2]
            for row, col in product(range(x.shape[-3]), range(x.shape[-2])):
                target_row, target_col = row + row_delta, col + col_delta
                if 0 <= target_row < x.shape[-3] and 0 <= target_col < x.shape[-2]:
                    result[..., direction, target_row, target_col, half] = source[..., row, col, 0]
    return result


@pytest.mark.parametrize("pre_norm,post_norm,use_bias", list(product([False, True], repeat=3)))
@pytest.mark.parametrize("entrypoint", ["forward", "__call__"])
def test_shift_primitives_glue(pre_norm, post_norm, use_bias, entrypoint):
    masks = torch.ones(2, 3, 4, 2, dtype=torch.float64)
    masks[0, 1, 1] = 0  # Neither input nor output includes this pixel.
    masks[1, 1, 2, 0] = 0  # Output-only pixels must still be included.
    masks[1, 2, 1, 1] = 0  # Input-only pixels must still be included.
    system = MultiTensorSystem(2, 2, 3, 4, task=None)
    inputs, weights = system.make_multitensor(), system.make_multitensor()
    generator = torch.Generator().manual_seed(17)
    expected = {}
    w1 = torch.tensor([[1.5, -0.5], [0.25, 2.0]], dtype=torch.float64)
    w2 = torch.tensor([[0.75, 0.5], [-1.0, 1.25]], dtype=torch.float64)
    b1 = torch.tensor([0.2, -0.3], dtype=torch.float64)
    b2 = torch.tensor([-0.4, 0.6], dtype=torch.float64)

    for dims in system:
        key = tuple(dims)
        x = torch.randn(system.shape(dims, extra_dim=2), generator=generator,
                        dtype=torch.float64)
        inputs[dims] = x
        if key not in ACTIVE_SHAPES:
            # Inactive leaves must pass through without trying to read weights.
            weights[dims] = None
            continue
        weights[dims] = [[w1, b1], [w2, b2]]
        projected = (reference_normalize(x) if pre_norm else x) @ w1
        if use_bias:
            projected = projected + b1
        shifted = reference_directions(projected, masks, has_colors=bool(dims[1]))
        if post_norm:
            shifted = reference_normalize(shifted)
        update = shifted @ w2
        if use_bias:
            update = update + b2
        expected[key] = x + update

    layer = ShiftPrimitives(multify)
    actual = getattr(layer, entrypoint)(inputs, weights, masks,
                                       pre_norm=pre_norm, post_norm=post_norm,
                                       use_bias=use_bias)
    assert actual.multitensor_system is system
    for dims in system:
        key = tuple(dims)
        if key in ACTIVE_SHAPES:
            torch.testing.assert_close(actual[dims], expected[key])
        else:
            assert actual[dims] is inputs[dims]
