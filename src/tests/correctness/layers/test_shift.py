"""Check primitive shifts; layer assembly is tested separately."""

import pytest
import torch

from compressarc.layers.shift.primitives import diagonal_shift_, shift_
from compressarc.layers.shift.morphological import TiedDirectionalConv

@pytest.fixture(
        params=[(2, 3), (0, 0), (4, 5)], ids=["interior", "top_left", "bottom_right"])
def pulse(request):
    position = request.param
    x = torch.zeros(1, 5, 6)
    x[0, position[0], position[1]] = 1
    return x, position


def expected_pulse(x, position, delta):
    expected = torch.zeros_like(x)
    row, col = (position[i] + delta[i] for i in range(2))
    if 0 <= row < x.shape[1] and 0 <= col < x.shape[2]:
        expected[0, row, col] = 1
    return expected


@pytest.mark.parametrize(
    "dim,reverse,delta",
    [(2, False, (0, 1)), (2, True, (0, -1)),
     (1, False, (1, 0)), (1, True, (-1, 0))],
    ids=["right", "left", "down", "up"],
)
def test_cardinal_shift(pulse, dim, reverse, delta):
    x, position = pulse
    # shift_ moves toward increasing indices; flipping gives the opposite direction.
    oriented = torch.flip(x, [dim]) if reverse else x
    actual = shift_(oriented, dim=dim, masks=None)
    if reverse:
        actual = torch.flip(actual, [dim])

    torch.testing.assert_close(actual, expected_pulse(x, position, delta), rtol=0, atol=0)

    

@pytest.mark.parametrize(
    "shift_amount,flip_columns,delta",
    [(1, False, (1, 1)), (1, True, (1, -1)),
     (-1, False, (-1, -1)), (-1, True, (-1, 1))],
    ids=["down_right", "down_left", "up_left", "up_right"],
)
def test_diagonal_shift(pulse, shift_amount, flip_columns, delta):
    x, position = pulse
    # The primitive uses one sign for both axes; flip columns for mixed directions.
    oriented = torch.flip(x, [2]) if flip_columns else x
    actual = diagonal_shift_(oriented, dim1=1, dim2=2, masks=None,
                             shift_amount=shift_amount)
    if flip_columns:
        actual = torch.flip(actual, [2])

    torch.testing.assert_close(actual, expected_pulse(x, position, delta), rtol=0, atol=0)

@pytest.mark.parametrize(
    "orbit,direction,delta",
    [(0, 0, (-1, 0)), (0, 1, (0, -1)),
     (0, 2, (1, 0)), (0, 3, (0, 1)),
     (1, 0, (1, 1)), (1, 1, (-1, 1)),
     (1, 2, (-1, -1)), (1, 3, (1, -1))],
    ids=["up", "left", "down", "right",
         "down_right", "up_right", "up_left", "down_left"],
)
def test_tied_directional_conv_matches_primitives(pulse, orbit, direction, delta):
    grid, _ = pulse
    # Different values in two batches also catch accidental batch mixing.
    grids = torch.cat([grid, 2 * grid], dim=0)
    x = torch.zeros(2, 2, 4, *grid.shape[-2:])
    x[:, orbit, direction] = grids

    # Orient the primitive toward the requested movement.
    flip_dims = [axis + 1 for axis, amount in enumerate(delta) if amount < 0]
    oriented = torch.flip(grids, flip_dims) if flip_dims else grids
    if orbit == 0:
        dim = 1 if delta[0] else 2
        shifted = shift_(oriented, dim=dim, masks=None)
    else:
        shifted = diagonal_shift_(oriented, dim1=1, dim2=2, masks=None)
    if flip_dims:
        shifted = torch.flip(shifted, flip_dims)

    expected = torch.zeros_like(x)
    expected[:, orbit, direction] = shifted
    actual = TiedDirectionalConv()(x)

    # Compare the entire output: untouched direction channels must stay zero.
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
