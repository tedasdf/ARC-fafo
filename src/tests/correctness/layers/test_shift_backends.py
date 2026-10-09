"""Unfold parity against learned conv2d, including the projected adapter."""
from itertools import product
import pytest
import torch

from compressarc.layers.shift.morphological import TiedDirectionalConv, UnfoldTiedDirectionalConv
from compressarc.model.layer_factory import LayerFactory
from compressarc.model.multitensor_systems import MultiTensorSystem, multify


@pytest.mark.parametrize("shape", [(1, 1), (1, 5), (5, 1), (3, 5), (5, 3)])
@pytest.mark.parametrize("layout", ["contiguous", "strided", "transposed"])
@pytest.mark.parametrize("learned", [False, True], ids=["initial", "learned"])
def test_unfold_core_outputs_and_gradients(shape, layout, learned):
    height, width = shape
    generator = torch.Generator().manual_seed(223)
    expected_model = TiedDirectionalConv().double()
    actual_model = UnfoldTiedDirectionalConv().double()
    if learned:
        with torch.no_grad():
            for parameter in expected_model.parameters():
                parameter.copy_(torch.randn(parameter.shape, generator=generator, dtype=torch.float64) * 0.2)
    actual_model.load_state_dict(expected_model.state_dict(), strict=True)
    if layout == "strided":
        values = torch.randn(2, 2, 4, height * 2 + 2, width * 2 + 2, generator=generator, dtype=torch.float64)
        view = lambda value: value[..., 1:1 + 2 * height:2, 1:1 + 2 * width:2]
    elif layout == "transposed":
        values = torch.randn(2, 2, 4, width, height, generator=generator, dtype=torch.float64)
        view = lambda value: value.transpose(-1, -2)
    else:
        values = torch.randn(2, 2, 4, height, width, generator=generator, dtype=torch.float64)
        view = lambda value: value
    expected_base = values.clone().requires_grad_()
    actual_base = values.clone().requires_grad_()
    expected, actual = expected_model(view(expected_base)), actual_model(view(actual_base))
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
    upstream = torch.randn(actual.shape, generator=generator, dtype=torch.float64)
    wanted = torch.autograd.grad((expected * upstream).sum(), [expected_base, *expected_model.parameters()])
    got = torch.autograd.grad((actual * upstream).sum(), [actual_base, *actual_model.parameters()])
    for observed, reference in zip(got, wanted):
        assert torch.isfinite(observed).all()
        torch.testing.assert_close(observed, reference, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("pre_norm,post_norm,use_bias", list(product([False, True], repeat=3)))
def test_unfold_factory_layer_matches_conv2d_masks_projections_and_gradients(pre_norm, post_norm, use_bias):
    generator = torch.Generator().manual_seed(227)
    system = MultiTensorSystem(2, 2, 3, 5, task=None)
    factory = LayerFactory()
    expected_layer = factory.create_shift("tied_conv", multify=multify).double()
    actual_layer = factory.create_shift("unfold", multify=multify).double()
    with torch.no_grad():
        for parameter in expected_layer.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator, dtype=torch.float64) * 0.2)
    actual_layer.load_state_dict(expected_layer.state_dict(), strict=True)
    inputs, weights = system.make_multitensor(), system.make_multitensor()
    targets, active = [], []
    for dims in system:
        inputs[dims] = torch.randn(system.shape(dims, extra_dim=4), generator=generator,
                                   dtype=torch.float64, requires_grad=True)
        if tuple(dims) not in ((1, 0, 1, 1, 1), (1, 1, 1, 1, 1)):
            weights[dims] = None
            continue
        active.append(dims)
        pair_shapes = ((4, 6), (6,), (6, 4), (4,))
        tensors = [torch.randn(shape, generator=generator, dtype=torch.float64, requires_grad=True)
                   for shape in pair_shapes]
        weights[dims] = [tensors[:2], tensors[2:]]
        targets.extend([inputs[dims], *tensors])
    masks = torch.randint(0, 2, (2, 3, 5, 2), generator=generator).double()
    flags = dict(pre_norm=pre_norm, post_norm=post_norm, use_bias=use_bias)
    expected, actual = expected_layer(inputs, weights, masks, **flags), actual_layer(inputs, weights, masks, **flags)
    expected_terms, actual_terms = [], []
    for dims in system:
        torch.testing.assert_close(actual[dims], expected[dims], rtol=1e-9, atol=1e-9)
        if dims not in active:
            assert actual[dims] is inputs[dims]
        else:
            upstream = torch.randn(actual[dims].shape, generator=generator, dtype=torch.float64)
            expected_terms.append((expected[dims] * upstream).sum())
            actual_terms.append((actual[dims] * upstream).sum())
    wanted = torch.autograd.grad(sum(expected_terms), [*targets, *expected_layer.parameters()], allow_unused=True)
    got = torch.autograd.grad(sum(actual_terms), [*targets, *actual_layer.parameters()], allow_unused=True)
    for observed, reference in zip(got, wanted):
        if reference is None:
            assert observed is None
        else:
            assert observed is not None and torch.isfinite(observed).all()
            torch.testing.assert_close(observed, reference, rtol=1e-8, atol=1e-8)
