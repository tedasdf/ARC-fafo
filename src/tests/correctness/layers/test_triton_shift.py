"""CUDA parity for direct Triton learned tied 3x3 convolution."""
import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("Triton shift requires CUDA", allow_module_level=True)

from compressarc.layers.shift.morphological import TiedDirectionalConv
from compressarc.layers.shift.triton_op import DepthwiseConvFunction, TritonTiedDirectionalConv


@pytest.mark.parametrize("shape", [(1, 1), (1, 5), (5, 1), (3, 5), (5, 3), (8, 8)])
@pytest.mark.parametrize("layout", ["contiguous", "strided", "transposed"])
@pytest.mark.parametrize("learned", [False, True])
def test_tied_core_outputs_and_canonical_kernel_gradients(shape, layout, learned):
    torch.manual_seed(239)
    h, w = shape
    reference = TiedDirectionalConv().cuda()
    actual_model = TritonTiedDirectionalConv().cuda()
    if learned:
        with torch.no_grad():
            for parameter in reference.parameters():
                parameter.normal_(std=0.2)
    actual_model.load_state_dict(reference.state_dict(), strict=True)
    if layout == "strided":
        values = torch.randn(2, 2, 4, 2 * h + 2, 2 * w + 2, device="cuda") * 0.2
        view = lambda value: value[..., 1:1 + 2 * h:2, 1:1 + 2 * w:2]
    elif layout == "transposed":
        values = torch.randn(2, 2, 4, w, h, device="cuda") * 0.2
        view = lambda value: value.transpose(-1, -2)
    else:
        values = torch.randn(2, 2, 4, h, w, device="cuda") * 0.2
        view = lambda value: value
    expected_x = values.clone().requires_grad_()
    actual_x = values.clone().requires_grad_()
    with torch.backends.cudnn.flags(allow_tf32=False):
        expected = reference(view(expected_x))
    actual = actual_model(view(actual_x))
    torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-6)
    upstream = torch.randn_like(actual).transpose(-1, -2).contiguous().transpose(-1, -2)
    with torch.backends.cudnn.flags(allow_tf32=False):
        wanted = torch.autograd.grad((expected * upstream).sum(), [expected_x, *reference.parameters()])
    got = torch.autograd.grad((actual * upstream).sum(), [actual_x, *actual_model.parameters()])
    for observed, reference_grad in zip(got, wanted):
        assert torch.isfinite(observed).all()
        torch.testing.assert_close(observed, reference_grad, rtol=3e-4, atol=8e-5)


@pytest.mark.parametrize("x_grad,k_grad", [(True, False), (False, True), (True, True)])
def test_repeat_backward_and_partial_gradients(x_grad, k_grad):
    x = torch.randn(2, 8, 3, 5, device="cuda", requires_grad=x_grad)
    kernel = torch.randn(8, 1, 3, 3, device="cuda", requires_grad=k_grad)
    output = DepthwiseConvFunction.apply(x, kernel)
    targets = [value for value in (x, kernel) if value.requires_grad]
    first = torch.autograd.grad(output.sum(), targets, retain_graph=True)
    second = torch.autograd.grad((2 * output).sum(), targets)
    for actual, previous in zip(second, first):
        torch.testing.assert_close(actual, 2 * previous, rtol=3e-4, atol=8e-5)


def test_validation():
    kernel = torch.zeros(8, 1, 3, 3, device="cuda")
    with pytest.raises(ValueError, match="nonempty"):
        DepthwiseConvFunction.apply(torch.zeros(2, 8, 0, 3, device="cuda"), kernel)
    with pytest.raises(ValueError, match="Kernel"):
        DepthwiseConvFunction.apply(torch.zeros(2, 8, 3, 5, device="cuda"), kernel[:, :, :2])
    with pytest.raises(ValueError, match="CUDA"):
        DepthwiseConvFunction.apply(torch.zeros(2, 8, 3, 5, device="cpu"), kernel)
    with pytest.raises(TypeError, match="float32"):
        DepthwiseConvFunction.apply(torch.zeros(2, 8, 3, 5, device="cuda", dtype=torch.float64), kernel)


@pytest.mark.parametrize("post_norm", [False, True])
def test_full_projected_layer_matches_conv2d(post_norm):
    from compressarc.model.layer_factory import LayerFactory
    from compressarc.model.multitensor_systems import MultiTensorSystem, multify

    torch.manual_seed(241)
    system = MultiTensorSystem(2, 2, 3, 5, task=None)
    factory = LayerFactory()
    reference = factory.create_shift("tied_conv", multify=multify).cuda()
    actual_layer = factory.create_shift("triton", multify=multify).cuda()
    with torch.no_grad():
        for parameter in reference.parameters():
            parameter.normal_(std=0.2)
    actual_layer.load_state_dict(reference.state_dict(), strict=True)
    inputs, weights = system.make_multitensor(), system.make_multitensor()
    targets, active = [], []
    for dims in system:
        inputs[dims] = (torch.randn(system.shape(dims, extra_dim=4), device="cuda") * 0.2).requires_grad_()
        if tuple(dims) not in ((1, 0, 1, 1, 1), (1, 1, 1, 1, 1)):
            weights[dims] = None
            continue
        active.append(dims)
        w1 = (torch.randn(4, 6, device="cuda") * 0.2).requires_grad_()
        w2 = (torch.randn(6, 4, device="cuda") * 0.2).requires_grad_()
        weights[dims] = [[w1, None], [w2, None]]
        targets.extend([inputs[dims], w1, w2])
    masks = torch.randint(0, 2, (2, 3, 5, 2), device="cuda").float()
    with torch.backends.cudnn.flags(allow_tf32=False):
        expected = reference(inputs, weights, masks, pre_norm=False, post_norm=post_norm)
    actual = actual_layer(inputs, weights, masks, pre_norm=False, post_norm=post_norm)
    wanted_terms, got_terms = [], []
    for dims in system:
        torch.testing.assert_close(actual[dims], expected[dims], rtol=3e-5, atol=5e-6)
        if dims not in active:
            assert actual[dims] is inputs[dims]
        else:
            upstream = torch.randn_like(actual[dims])
            wanted_terms.append((expected[dims] * upstream).sum())
            got_terms.append((actual[dims] * upstream).sum())
    with torch.backends.cudnn.flags(allow_tf32=False):
        wanted = torch.autograd.grad(sum(wanted_terms), [*targets, *reference.parameters()])
    got = torch.autograd.grad(sum(got_terms), [*targets, *actual_layer.parameters()])
    for observed, reference_grad in zip(got, wanted):
        assert torch.isfinite(observed).all()
        torch.testing.assert_close(observed, reference_grad, rtol=5e-4, atol=1e-4)
