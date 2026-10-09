"""Projected-D4 pilot equation, adapter gradients, and optimizer registration."""
import pytest
import torch

from config import load_config
from preprocessing import Task
from compressarc.model.layer_factory import LayerFactory
from compressarc.model.model import ARCCompressor
from compressarc.model.multitensor_systems import MultiTensorSystem, multify
from compressarc.train import take_step


@pytest.mark.parametrize("pre_norm", [False, True])
def test_projected_d4_matches_pilot_equation_and_gradients(pre_norm):
    generator = torch.Generator().manual_seed(53)
    system = MultiTensorSystem(2, 2, 2, 3, task=None)
    layer = LayerFactory().create_direction_share("d4", multify=multify, n_layers=2).double()
    inputs, weights = system.make_multitensor(), system.make_multitensor()
    targets, directional_dims = [], []
    expected = {}
    matrix = layer.models[1].weight()
    for dims in system:
        x = torch.randn(system.shape(dims, extra_dim=3), generator=generator,
                        dtype=torch.float64, requires_grad=True)
        inputs[dims] = x
        if not dims[2]:
            weights[dims] = None
            expected[tuple(dims)] = x
            continue
        directional_dims.append(dims)
        w1 = torch.randn(3, 8, generator=generator, dtype=torch.float64, requires_grad=True)
        w2 = torch.randn(8, 3, generator=generator, dtype=torch.float64, requires_grad=True)
        weights[dims] = [[w1, None], [w2, None]]
        targets.extend([x, w1, w2])
        z = x
        if pre_norm:
            axes = tuple(range(x.ndim - 1))
            centered = x - x.mean(dim=axes)
            z = centered / torch.sqrt(1e-8 + centered.square().mean(dim=axes))
        projected = z @ w1
        axis = sum(dims[:2])
        mixed = torch.einsum("ij,j...->i...", matrix, projected.movedim(axis, 0)).movedim(0, axis)
        expected[tuple(dims)] = x + mixed @ w2
    actual = layer(inputs, weights, pre_norm=pre_norm, layer_index=1)
    actual_terms, expected_terms = [], []
    for dims in system:
        torch.testing.assert_close(actual[dims], expected[tuple(dims)], rtol=1e-10, atol=1e-10)
        if not dims[2]:
            assert actual[dims] is inputs[dims]
        else:
            upstream = torch.randn(actual[dims].shape, generator=generator, dtype=torch.float64)
            actual_terms.append((actual[dims] * upstream).sum())
            expected_terms.append((expected[tuple(dims)] * upstream).sum())
    targets.append(layer.models[1].theta)
    wanted = torch.autograd.grad(sum(expected_terms), targets)
    got = torch.autograd.grad(sum(actual_terms), [*targets, layer.models[0].theta], allow_unused=True)
    assert got[-1] is None
    for observed, reference in zip(got[:-1], wanted):
        assert observed is not None and torch.isfinite(observed).all()
        torch.testing.assert_close(observed, reference, rtol=1e-9, atol=1e-9)
    assert layer.models[0].theta is not layer.models[1].theta


def test_projected_d4_model_registers_and_updates_projections_and_mixer():
    config = load_config(overrides=[
        "model.direction_share_implementation=d4", "model.n_layers=1",
        "model.channel_dim_shared=2", "model.channel_dim_grid=2",
        "model.share_up_dim=2", "model.share_down_dim=2", "model.decoding_dim=2",
        "model.softmax_dim=2", "model.cummax_dim=2", "model.shift_dim=2", "model.nonlinear_dim=2",
    ])
    task = Task("projection_test", {
        "train": [{"input": [[0, 1], [1, 0]], "output": [[0, 1], [1, 0]]}],
        "test": [{"input": [[1, 0], [0, 1]]}],
    }, None)
    model = ARCCompressor(task, config.model)
    dims = (1, 0, 1, 1, 1)
    projection = model.direction_projection_weights[0]
    assert model.layer_weights[0]["direction_share"] is projection
    w1, w2 = projection[dims][0][0], projection[dims][1][0]
    theta = model.direction_share.models[0].theta
    assert w1.shape == (2, 8) and w2.shape == (8, 2)
    registered = {id(weight) for weight in model.weights_list}
    assert len(registered) == len(model.weights_list)
    assert {id(w1), id(w2), id(theta)} <= registered
    optimizer = torch.optim.Adam(model.weights_list, lr=0.01)
    metrics = take_step(task, model, optimizer, 0, config.training)
    assert torch.isfinite(torch.tensor(metrics["loss"]))
    for value in (w1, w2, theta):
        assert value.grad is not None and torch.isfinite(value.grad).all()
    # A task can give a component zero gradient. Use a controlled layer loss
    # to verify the registered projection matrices and mixer really update.
    generator = torch.Generator().manual_seed(71)
    inputs = task.multitensor_system.make_multitensor()
    for component in task.multitensor_system:
        inputs[component] = torch.randn(
            task.multitensor_system.shape(component, extra_dim=model.channel_dim_fn(component)),
            generator=generator,
        )
    optimizer.zero_grad()
    before = [value.detach().clone() for value in (w1, w2, theta)]
    output = model.direction_share(inputs, projection, pre_norm=True, layer_index=0)
    output[dims].square().sum().backward()
    optimizer.step()
    for old, value in zip(before, (w1, w2, theta)):
        assert value.grad is not None and torch.isfinite(value.grad).all()
        assert torch.count_nonzero(value.grad) > 0
        assert not torch.equal(old, value)
