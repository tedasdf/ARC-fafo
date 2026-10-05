"""Verify D4 direction mixing and its CompressARC axis adapter."""

from itertools import product

import pytest
import torch

from compressarc.layers.direction_share.morphological import (
    D4DirectionShare, compressarc_direction_share,
)


def model_with_distinct_orbits():
    model = D4DirectionShare().double()
    with torch.no_grad():
        model.theta.copy_(torch.arange(1, 11, dtype=torch.float64) / 10)
    return model


@pytest.mark.parametrize('rotation,reflection', list(product(range(4), [False, True])))
def test_d4_direction_share_equivariance(rotation, reflection):
    model = model_with_distinct_orbits()
    x = torch.arange(2 * 8 * 3 * 5, dtype=torch.float64).reshape(2, 8, 3, 5)
    directions = torch.arange(8)
    permutation = ((-directions if reflection else directions) + 2 * rotation) % 8
    actual = model(x[:, permutation])
    expected = model(x)[:, permutation]
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize('input_direction', range(8))
def test_d4_direction_share_orbit_weights(input_direction):
    model = model_with_distinct_orbits()
    x = torch.zeros(1, 8, 2, 3, dtype=torch.float64)
    x[:, input_direction] = 1
    expected = torch.zeros_like(x)
    for output_direction in range(8):
        distance = min((input_direction - output_direction) % 8,
                       (output_direction - input_direction) % 8)
        if output_direction % 2 == input_direction % 2:
            orbit = (3 if output_direction % 2 else 0) + distance // 2
        else:
            orbit = (8 if output_direction % 2 else 6) + (distance - 1) // 2
        expected[:, output_direction] = (orbit + 1) / 10
    torch.testing.assert_close(model(x), expected)


def test_d4_direction_share_parameter_gradients():
    model = model_with_distinct_orbits()
    x = torch.ones(2, 8, 3, 5, dtype=torch.float64, requires_grad=True)
    model(x).sum().backward()
    assert model.theta.numel() == 10
    assert model.theta.grad.shape == (10,)
    assert torch.isfinite(model.theta.grad).all()
    assert (model.theta.grad > 0).all()
    assert torch.isfinite(x.grad).all()


@pytest.mark.parametrize('dims', list(product([0, 1], repeat=5)))
def test_d4_adapter_axes_residual_and_passthrough(dims):
    model = model_with_distinct_orbits()
    lengths = [2, 3, 8, 4, 5]
    shape = [length for flag, length in zip(dims, lengths) if flag] + [2]
    x = torch.arange(torch.tensor(shape).prod().item(), dtype=torch.float64).reshape(shape) / 100
    actual = compressarc_direction_share(dims, x, model)
    if not dims[2]:
        assert actual is x
        return
    axis = sum(dims[:2])
    directions_first = x.movedim(axis, 0)
    # Contract over the direction dimension directly; no adapter axis manipulation.
    mixed = torch.tensordot(model.weight(), directions_first, dims=([1], [0]))
    expected = x + mixed.movedim(0, axis)
    torch.testing.assert_close(actual, expected)



def test_factory_d4_multitensor_wrapper_and_optimizer():
    from compressarc.model.layer_factory import LayerFactory
    from compressarc.model.multitensor_systems import MultiTensorSystem, multify
    from compressarc.layers.helper import normalize

    system = MultiTensorSystem(2, 2, 3, 4, task=None)
    inputs = system.make_multitensor()
    for dims in system:
        inputs[dims] = torch.randn(system.shape(dims, extra_dim=2), dtype=torch.float64)
    layer = LayerFactory().create_direction_share('d4', multify=multify, n_layers=2).double()
    assert layer.models[0].theta is not layer.models[1].theta
    for index in range(2):
        actual = layer(inputs, None, pre_norm=True, use_bias=False, layer_index=index)
        for dims in system:
            if not dims[2]:
                assert actual[dims] is inputs[dims]
            else:
                z = normalize(inputs[dims])
                mixed = compressarc_direction_share(dims, z, layer.models[index]) - z
                torch.testing.assert_close(actual[dims], inputs[dims] + mixed)
    optimizer = torch.optim.SGD(layer.parameters(), lr=0.1)
    before = [m.theta.detach().clone() for m in layer.models]
    loss = sum(layer(inputs, layer_index=i, pre_norm=False)[[1, 0, 1, 1, 1]].square().sum()
               for i in range(2))
    loss.backward()
    optimizer.step()
    assert all(not torch.equal(old, m.theta) for old, m in zip(before, layer.models))
