"""Benchmark primitive operations and complete shift/cummax layers.

All timings use eager float32 execution without W&B, PCA, or internal timers.
CUDA results include synchronized wall latency and CUDA-event elapsed latency.
"""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import statistics
import time

import torch

from compressarc.layers.cummax.morphological import MorphologicalMax as ReferenceLSE
from compressarc.layers.cummax.optimisation import MorphologicalMax as CurrentLSE
from compressarc.layers.cummax.primitives import cummax_, diagonal_cummax_
from compressarc.layers.shift.primitives import shift_, diagonal_shift_
from compressarc.model.layer_factory import LayerFactory
from compressarc.model.multitensor_systems import MultiTensorSystem, multify


def percentile(values, q):
    values = sorted(values)
    position = (len(values) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def measure(fn, inputs, parameters, *, device, backward, warmup, repeats):
    cuda = device.type == 'cuda'

    def reset():
        for tensor in [*inputs, *parameters]:
            tensor.grad = None

    def step():
        with torch.set_grad_enabled(backward):
            outputs = fn()
            if backward:
                sum(output.sum() for output in outputs).backward()

    with torch.no_grad():
        if not all(torch.isfinite(output).all().item() for output in fn()):
            raise RuntimeError('Non-finite benchmark output')
    for _ in range(warmup):
        reset()
        step()
    wall_ms, event_ms = [], []
    for _ in range(repeats):
        reset()  # Gradient reset is outside the timed region.
        if cuda:
            torch.cuda.synchronize(device)
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        started = time.perf_counter()
        if cuda:
            start.record()
        step()
        if cuda:
            end.record()
            end.synchronize()
        wall_ms.append((time.perf_counter() - started) * 1000)
        if cuda:
            event_ms.append(start.elapsed_time(end))
    reset()
    return {
        'median_wall_ms': statistics.median(wall_ms),
        'p90_wall_ms': percentile(wall_ms, 0.9),
        'min_wall_ms': min(wall_ms),
        'median_cuda_ms': statistics.median(event_ms) if event_ms else None,
        'wall_samples_ms': wall_ms,
        'cuda_samples_ms': event_ms,
    }


def primitive_cases(batch, colors, height, width, features, device):
    # Treat colors and feature halves as independent grids, as the adapters do.
    packed_batch = batch * colors * (features // 2)
    x = torch.randn(packed_batch, height, width, device=device, requires_grad=True)
    mask = torch.ones_like(x)
    yield 'shift', 'cardinal', lambda: [shift_(x, 2, None)], [x], []
    yield 'shift', 'diagonal', lambda: [diagonal_shift_(x, 1, 2, None)], [x], []
    yield 'cummax', 'cardinal', lambda: [cummax_(x, 2, mask)], [x], []
    yield 'cummax', 'diagonal', lambda: [diagonal_cummax_(x, 1, 2, mask)], [x], []
    for name, cls in [('lse_reference', ReferenceLSE), ('lse_current', CurrentLSE)]:
        model = cls(height, width, timing=False).to(device)
        yield name, 'cardinal', lambda m=model: [m._axis_lse(x)], [x], list(model.parameters())
        yield name, 'diagonal', lambda m=model: [m._diagonal_lse(x)], [x], list(model.parameters())


def layer_cases(batch, colors, height, width, features, residual_features, device):
    system = MultiTensorSystem(batch, colors, height, width, task=None)
    inputs, weights = system.make_multitensor(), system.make_multitensor()
    leaves, projection_parameters = [], []
    active = [(1, 0, 1, 1, 1), (1, 1, 1, 1, 1)]
    # These are exactly the two active shapes supported by the migrated layers.
    for dims in active:
        x = torch.randn(system.shape(dims, extra_dim=residual_features),
                        device=device, requires_grad=True)
        w1 = (torch.randn(residual_features, features, device=device) /
              residual_features**0.5).requires_grad_()
        w2 = (torch.randn(features, residual_features, device=device) /
              features**0.5).requires_grad_()
        inputs[dims] = x
        weights[dims] = [[w1, None], [w2, None]]
        leaves.append(x)
        projection_parameters.extend([w1, w2])
    masks = torch.ones(batch, height, width, 2, device=device)
    factory = LayerFactory()
    for name, implementations in [('shift', ['primitives', 'tied_conv']),
                                  ('cummax', ['primitives', 'd4', 'optimised'])]:
        for implementation in implementations:
            options = dict(height=height, width=width, timing=False) if name == 'cummax' and implementation != 'primitives' else {}
            layer = factory.create(name, implementation, multify=multify, **options)
            parameters = list(projection_parameters)
            if isinstance(layer, torch.nn.Module):
                layer.to(device)
                parameters.extend(layer.parameters())

            def call(layer=layer):
                result = layer(inputs, weights, masks, pre_norm=False, post_norm=True, use_bias=False)
                return [result[dims] for dims in active]

            yield name, implementation, call, leaves, parameters


def benchmark(*, device='cuda', shapes=((8, 8), (16, 16)), batch=2, colors=2,
              features=4, residual_features=8, warmup=2, repeats=5, threads=1,
              scopes=('primitive', 'layer'), modes=('forward', 'forward_backward')):
    if min(batch, colors, features, residual_features, repeats, threads) < 1 or warmup < 0:
        raise ValueError('Sizes, repetitions and threads must be positive; warmup must be nonnegative')
    if features % 2:
        raise ValueError('Directional feature count must be even')
    if any(min(shape) < 1 for shape in shapes):
        raise ValueError('Spatial sizes must be positive')
    device = torch.device(device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is unavailable')
    torch.set_num_threads(threads)
    torch.manual_seed(42)
    results = []
    metadata = dict(
        created_at=datetime.now(timezone.utc).isoformat(), pytorch=str(torch.__version__),
        device=str(device), device_name=torch.cuda.get_device_name(device) if device.type == 'cuda' else platform.processor(),
        cuda_version=torch.version.cuda, dtype='float32', threads=threads,
        batch=batch, colors=colors, features=features, residual_features=residual_features,
        warmup=warmup, repeats=repeats, execution='eager; no autocast or compilation',
        layer_scope='Both active multitensor shapes, projections, post-normalization, residual and adapter',
        primitive_scope='One cardinal/diagonal operation on batch*colors*(features/2) grids',
        note='LSE is smooth morphology; primitive cummax also rescales outputs. Latency comparison does not imply identical semantics.',
    )
    for height, width in shapes:
        for scope in scopes:
            cases = primitive_cases(batch, colors, height, width, features, device) if scope == 'primitive' else layer_cases(batch, colors, height, width, features, residual_features, device)
            for layer, implementation, fn, inputs, parameters in cases:
                for mode in modes:
                    timing = measure(fn, inputs, parameters, device=device,
                                     backward=mode == 'forward_backward', warmup=warmup, repeats=repeats)
                    row = dict(scope=scope, layer=layer, implementation=implementation,
                               height=height, width=width, mode=mode, **timing)
                    results.append(row)
                    print(f'{height}x{width} {scope:9} {layer:13} {implementation:10} {mode:16} {timing["median_wall_ms"]:9.3f} ms', flush=True)
    return dict(metadata=metadata, results=results)


def write_results(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')
    fields = [key for key in data['results'][0] if not key.endswith('samples_ms')]
    with path.with_suffix('.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(data['results'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--shapes', nargs='+', default=['8x8', '16x16'])
    for name, default in [('batch', 2), ('colors', 2), ('features', 4), ('residual-features', 8), ('warmup', 2), ('repeats', 5), ('threads', 1)]:
        parser.add_argument('--' + name, type=int, default=default)
    parser.add_argument('--scope', choices=['primitive', 'layer', 'both'], default='both')
    parser.add_argument('--mode', choices=['forward', 'forward_backward', 'both'], default='both')
    parser.add_argument('--output', type=Path, default=Path('outputs/benchmarks/shift_cummax.json'))
    args = parser.parse_args()
    shapes = [tuple(map(int, value.lower().split('x'))) for value in args.shapes]
    if any(len(shape) != 2 for shape in shapes):
        parser.error('Shapes must have the form HxW')
    data = benchmark(device=args.device, shapes=shapes, batch=args.batch, colors=args.colors,
                     features=args.features, residual_features=args.residual_features,
                     warmup=args.warmup, repeats=args.repeats, threads=args.threads,
                     scopes=('primitive', 'layer') if args.scope == 'both' else (args.scope,),
                     modes=('forward', 'forward_backward') if args.mode == 'both' else (args.mode,))
    write_results(data, args.output)
    print(f'Results: {args.output.resolve()}', flush=True)


if __name__ == '__main__':
    main()
