"""GPU checks for the Triton diagonal LSE forward, dx and dk."""
import pytest
import torch

pytest.importorskip('triton')
if not torch.cuda.is_available():
    pytest.skip('Triton LSE requires CUDA', allow_module_level=True)

from compressarc.layers.cummax.triton_op import DiagonalLSEFunction, FusedDiagonalLSE


def reference(x, kernel, tau):
    _, height, width = x.shape
    result = torch.empty_like(x)
    for row in range(height):
        for col in range(width):
            u = torch.arange(min(row, col) + 1, device=x.device)
            scores = x[:, row - u, col - u] + kernel[u]
            result[:, row, col] = tau * torch.logsumexp(scores / tau, dim=-1)
    return result


@pytest.mark.parametrize('shape', [(1, 1), (1, 7), (7, 1), (3, 5), (5, 3), (8, 8), (7, 16)])
@pytest.mark.parametrize('tau', [0.03, 0.1, 0.7])
@pytest.mark.parametrize('layout', ['contiguous', 'strided', 'transposed'])
def test_forward_input_and_kernel_gradients(shape, tau, layout):
    height, width = shape
    torch.manual_seed(71)
    if layout == 'strided':
        base = torch.randn(3, height * 2 + 2, width * 2 + 2, device='cuda') * 0.2
        view = lambda x: x[1:, 1:1 + height * 2:2, 1:1 + width * 2:2]
    elif layout == 'transposed':
        base = torch.randn(2, width, height, device='cuda') * 0.2
        view = lambda x: x.transpose(-1, -2)
    else:
        base = torch.randn(2, height, width, device='cuda') * 0.2
        view = lambda x: x
    reference_base = base.clone().requires_grad_()
    triton_base = base.clone().requires_grad_()
    # Include an unused tail; its gradient must remain exactly zero.
    kernel = torch.randn(2 * (min(height, width) + 3), device='cuda') * 0.15
    reference_kernel = kernel.clone()[::2].detach().requires_grad_()
    triton_kernel = kernel.clone()[::2].detach().requires_grad_()
    expected = reference(view(reference_base), reference_kernel, tau)
    actual = DiagonalLSEFunction.apply(view(triton_base), triton_kernel, tau)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    dy = torch.randn_like(actual).transpose(-1, -2).contiguous().transpose(-1, -2)
    expected.backward(dy)
    actual.backward(dy)
    torch.testing.assert_close(triton_base.grad, reference_base.grad, rtol=3e-4, atol=3e-5)
    torch.testing.assert_close(triton_kernel.grad, reference_kernel.grad, rtol=3e-4, atol=8e-5)
    assert torch.count_nonzero(triton_kernel.grad[min(height, width):]) == 0
    assert torch.isfinite(triton_base.grad).all()
    assert torch.isfinite(triton_kernel.grad).all()


@pytest.mark.parametrize('x_grad,k_grad', [(True, False), (False, True), (True, True)])
def test_repeat_backward_zeroes_atomic_buffer(x_grad, k_grad):
    x = torch.randn(2, 3, 5, device='cuda', requires_grad=x_grad)
    kernel = torch.randn(5, device='cuda', requires_grad=k_grad)
    actual = DiagonalLSEFunction.apply(x, kernel, 0.3)
    targets = [p for p in [x, kernel] if p.requires_grad]
    first = torch.autograd.grad(actual.sum(), targets, retain_graph=True)
    second = torch.autograd.grad((actual * 2).sum(), targets)
    for got, previous in zip(second, first):
        torch.testing.assert_close(got, 2 * previous, rtol=3e-4, atol=8e-5)


def test_module_optimizer_and_unused_tail():
    layer = FusedDiagonalLSE(8, tau=0.1).cuda()
    x = torch.randn(2, 3, 5, device='cuda', requires_grad=True)
    before = layer.kernel.detach().clone()
    optimizer = torch.optim.SGD(layer.parameters(), lr=0.01)
    layer(x).mean().backward()
    assert x.grad.shape == x.shape
    assert layer.kernel.grad.shape == (8,)
    assert torch.count_nonzero(layer.kernel.grad[3:]) == 0
    optimizer.step()
    assert not torch.equal(before[:3], layer.kernel[:3])
    torch.testing.assert_close(before[3:], layer.kernel[3:])


@pytest.mark.parametrize('tau', [0.0, -0.1, float('inf'), float('nan')])
def test_invalid_temperature(tau):
    with pytest.raises(ValueError, match='tau'):
        DiagonalLSEFunction.apply(torch.zeros(1, 2, 3, device='cuda'),
                                  torch.zeros(2, device='cuda'), tau)


def test_validation():
    layer = FusedDiagonalLSE(2).cuda()
    with pytest.raises(ValueError, match='max_length'):
        layer(torch.zeros(1, 3, 4, device='cuda'))
    with pytest.raises(TypeError, match='float32'):
        DiagonalLSEFunction.apply(torch.zeros(1, 2, 3, device='cuda', dtype=torch.float64),
                                  torch.zeros(2, device='cuda'), 0.1)

@pytest.mark.parametrize('shape', [(1, 1), (1, 7), (7, 1), (3, 5), (5, 3), (8, 8)])
@pytest.mark.parametrize('tau', [0.03, 0.1, 0.7])
@pytest.mark.parametrize('transposed', [False, True], ids=['contiguous', 'transposed'])
def test_triton_diagonal_matches_optimisation_output_and_gradients(shape, tau, transposed):
    from compressarc.layers.cummax.optimisation import MorphologicalMax

    height, width = shape
    torch.manual_seed(29)
    if transposed:
        values = (torch.randn(2, width, height, device='cuda') * 0.2).transpose(-1, -2)
    else:
        values = torch.randn(2, height, width, device='cuda') * 0.2
    x_reference = values.detach().clone().requires_grad_()
    x_triton = values.detach().clone().requires_grad_()

    reference_model = MorphologicalMax(height, width, tau=tau, timing=False).cuda()
    triton_model = FusedDiagonalLSE(min(height, width), tau=tau).cuda()
    with torch.no_grad():
        reference_model.diagonal_kernel.normal_(std=0.15)
        triton_model.kernel.copy_(reference_model.diagonal_kernel)

    expected = reference_model._diagonal_lse(x_reference)
    actual = triton_model(x_triton)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)

    # Use identical, nonuniform  plus gradients to check the full backward map.
    upstream = torch.randn_like(expected)
    expected.backward(upstream)
    actual.backward(upstream)
    assert x_reference.grad is not None and x_triton.grad is not None
    assert reference_model.diagonal_kernel.grad is not None and triton_model.kernel.grad is not None
    torch.testing.assert_close(x_triton.grad, x_reference.grad, rtol=3e-4, atol=3e-5)
    torch.testing.assert_close(triton_model.kernel.grad, reference_model.diagonal_kernel.grad,
                               rtol=3e-4, atol=8e-5)
