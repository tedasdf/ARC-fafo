"""Direction checks and regression checks for both LSE implementations."""

import pytest
import torch

from compressarc.layers.cummax.optimisation import MorphologicalMax
from compressarc.layers.cummax.morphological import MorphologicalMax as MorphologicalMaxPreoptimized

SHAPES = [(1, 1), (1, 5), (5, 1), (3, 5), (5, 3), (4, 4)]
DIRECTIONS = [(0, 1), (1, 1), (-1, 0), (-1, 1),
              (0, -1), (-1, -1), (1, 0), (1, -1)]


def reference_direction(x, kernel, tau, delta):
    # Explicit predecessor coordinates, independent of rotations and strided views.
    dr, dc = delta
    result = torch.empty_like(x)
    for row in range(x.shape[-2]):
        for col in range(x.shape[-1]):
            scores = []
            r, c, offset = row, col, 0
            while 0 <= r < x.shape[-2] and 0 <= c < x.shape[-1]:
                scores.append(x[:, r, c] + kernel[offset])
                r, c, offset = r - dr, c - dc, offset + 1
            result[:, row, col] = tau * torch.logsumexp(torch.stack(scores, -1) / tau, -1)
    return result


def initialize(model):
    with torch.no_grad():
        model.kernel.copy_(torch.linspace(-0.2, 0.4, model.kernel.numel()))
        model.diagonal_kernel.copy_(torch.linspace(0.3, -0.1, model.diagonal_kernel.numel()))
    return model


@pytest.mark.parametrize('implementation', [MorphologicalMaxPreoptimized, MorphologicalMax],
                         ids=['preoptimized', 'current'])
@pytest.mark.parametrize('shape', SHAPES)
@pytest.mark.parametrize('tau', [0.1, 0.7])
def test_lse_all_directions(implementation, shape, tau):
    height, width = shape
    generator = torch.Generator().manual_seed(29)
    x = torch.randn(2, height, width, generator=generator, dtype=torch.float64)
    model = initialize(implementation(height, width, tau=tau).double())
    actual = model(x)
    assert actual.shape == (2, 8, height, width)
    for direction, delta in enumerate(DIRECTIONS):
        kernel = model.kernel if direction % 2 == 0 else model.diagonal_kernel
        expected = reference_direction(x, kernel, tau, delta)
        torch.testing.assert_close(actual[:, direction], expected)
        torch.testing.assert_close(model.direction(x, direction), expected)


@pytest.mark.parametrize('shape', SHAPES)
@pytest.mark.parametrize('rotation', range(4))
@pytest.mark.parametrize('sliced', [False, True], ids=['offset', 'strided'])
def test_current_matches_preoptimized_outputs_and_gradients(shape, rotation, sliced):
    height, width = shape
    generator = torch.Generator().manual_seed(42)
    base = torch.randn(3, 2 * height + 2, 2 * width + 2,
                       generator=generator, dtype=torch.float64, requires_grad=True)
    if sliced:
        x = base[1:, 1:1 + 2 * height:2, 1:1 + 2 * width:2]
    else:
        x = base[1:, 1:1 + height, 1:1 + width]
    x = torch.rot90(x, rotation, (-2, -1))
    reference = initialize(MorphologicalMaxPreoptimized(height, width, tau=0.3).double())
    current = MorphologicalMax(height, width, tau=0.3).double()
    current.load_state_dict(reference.state_dict(), strict=True)
    expected, actual = reference(x), current(x)
    torch.testing.assert_close(actual, expected)
    upstream = torch.randn(actual.shape, generator=generator, dtype=torch.float64)
    expected_grad = torch.autograd.grad((expected * upstream).sum(),
                                       (base, reference.kernel, reference.diagonal_kernel))
    actual_grad = torch.autograd.grad((actual * upstream).sum(),
                                     (base, current.kernel, current.diagonal_kernel))
    for got, want in zip(actual_grad, expected_grad):
        assert torch.isfinite(got).all()
        torch.testing.assert_close(got, want)



@pytest.mark.parametrize('implementation', [MorphologicalMaxPreoptimized, MorphologicalMax])
def test_timing_disabled_by_default(implementation, monkeypatch):
    from compressarc.layers import helper

    def unexpected_timer():
        raise AssertionError('Timing disabled: clock must not be read')

    monkeypatch.setattr(helper.time, 'perf_counter', unexpected_timer)
    model = implementation(3, 4)
    model(torch.zeros(1, 3, 4))
    assert model.timings == {}


@pytest.mark.parametrize('implementation', [MorphologicalMaxPreoptimized, MorphologicalMax])
def test_timing_flag_preserves_outputs_and_records_calls(implementation):
    model = implementation(3, 4, timing=True)
    plain = implementation(3, 4)
    plain.load_state_dict(model.state_dict())
    x = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4)
    torch.testing.assert_close(model(x), plain(x))
    for name in ['phi', 'diagonal_phi']:
        assert model.timings[f'timing/{name}_calls'] == 4
        assert model.timings[f'timing/{name}_forward_seconds'] >= 0
    model.timings.clear()
    model.timing = False
    model(x)
    assert model.timings == {}

