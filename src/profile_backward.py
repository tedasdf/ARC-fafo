"""Measure isolated component backward latency using real inputs and upstream gradients.

No optimiser updates, logging, profiler or forward work occurs inside timed VJPs.
These isolated timings are diagnostic and must not be summed as whole-model time.
"""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import statistics
import time
from unittest.mock import patch

import numpy as np
import torch

from config import load_config
from preprocessing import preprocess_tasks
from compressarc.model.model import ARCCompressor
from compressarc.model.multitensor_systems import MultiTensor
from compressarc.train import take_step
from compressarc.train.device_dispatch import runtime_device_dispatch
from compressarc.train.optimizer import create_optimizer

COMPONENTS = ('share_up', 'share_down', 'direction_share', 'shift', 'cummax')


def tensors(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, MultiTensor):
        for dims in value.multitensor_system:
            yield from tensors(value[dims])
    elif isinstance(value, dict):
        for item in value.values():
            yield from tensors(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from tensors(item)


def unique_tensors(value):
    return list({id(x): x for x in tensors(value)}.values())


def detach_inputs(value, memo=None):
    """Detach at the component boundary, preserving aliases, devices and strides."""
    memo = {} if memo is None else memo
    if isinstance(value, torch.Tensor):
        if id(value) not in memo:
            memo[id(value)] = value.detach().requires_grad_(value.requires_grad)
        return memo[id(value)]
    if isinstance(value, MultiTensor):
        result = value.multitensor_system.make_multitensor()
        for dims in value.multitensor_system:
            result[dims] = detach_inputs(value[dims], memo)
        return result
    if isinstance(value, list):
        return [detach_inputs(x, memo) for x in value]
    if isinstance(value, tuple):
        return tuple(detach_inputs(x, memo) for x in value)
    if isinstance(value, dict):
        return {k: detach_inputs(v, memo) for k, v in value.items()}
    return value


@contextmanager
def capture_components(model, selected=COMPONENTS):
    """Capture boundary inputs and real loss gradients during one ordinary step."""
    records, patches, handles, counts = [], [], [], {}
    def wrap(name, fn):
        def captured(*args, **kwargs):
            index = counts.get(name, 0)
            counts[name] = index + 1
            detached = detach_inputs(args[0])
            output = fn(*args, **kwargs)
            outputs = [x for x in unique_tensors(output) if x.requires_grad]
            record = dict(name=f'depth_{index}/{name}', call=fn,
                          args=(detached, *args[1:]), kwargs=kwargs,
                          upstream=[None] * len(outputs))
            for i, tensor in enumerate(outputs):
                def save(gradient, record=record, i=i):
                    record['upstream'][i] = gradient.detach().clone()
                handles.append(tensor.register_hook(save))
            records.append(record)
            return output
        return captured
    try:
        for name in selected:
            owner = model.share if name in ('share_up', 'share_down') else model
            attribute = name
            fn = getattr(owner, attribute)
            replacement = patch.object(owner, attribute, wrap(name, fn))
            replacement.start()
            patches.append(replacement)
        yield records
    finally:
        for handle in handles:
            handle.remove()
        for replacement in reversed(patches):
            replacement.stop()


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def measure_backward(call, device):
    """Elapsed completion time; CUDA event spans include CPU dispatch gaps."""
    synchronize(device)
    start = end = None
    if device.type == 'cuda':
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
    began = time.perf_counter()
    call()
    if end is not None:
        end.record()
    synchronize(device)
    wall = time.perf_counter() - began
    return wall, start.elapsed_time(end) / 1000 if start is not None else None


def summarize(samples):
    wall = [s[0] for s in samples]
    stream = [s[1] for s in samples if s[1] is not None]
    return dict(samples=len(wall), wall_mean_seconds=statistics.mean(wall),
                wall_median_seconds=statistics.median(wall),
                wall_min_seconds=min(wall), wall_max_seconds=max(wall),
                cuda_stream_mean_seconds=statistics.mean(stream) if stream else None)


def benchmark_component(record, parameters, device, warmup=5, steps=20):
    inputs = unique_tensors(record['args'][0])
    samples = []
    for iteration in range(warmup + steps):
        for tensor in [*parameters, *inputs]:
            tensor.grad = None
        # Fresh graph each time. Forward, clears and barriers are outside timing.
        output = record['call'](*record['args'], **record['kwargs'])
        outputs = [x for x in unique_tensors(output) if x.requires_grad]
        if len(outputs) != len(record['upstream']):
            raise RuntimeError('Component output structure changed during replay')
        pairs = [(x, g) for x, g in zip(outputs, record['upstream']) if g is not None]
        if not pairs:
            raise RuntimeError('No observed upstream gradient for '+record['name'])
        elapsed = measure_backward(lambda: torch.autograd.backward(
            [x for x, _ in pairs], [g for _, g in pairs]), device)
        if iteration >= warmup:
            samples.append(elapsed)
        del output, outputs, pairs
    return dict(name=record['name'], input_shapes=[list(x.shape) for x in inputs],
                input_strides=[list(x.stride()) for x in inputs],
                active_outputs=sum(g is not None for g in record['upstream']),
                output_count=len(record['upstream']), **summarize(samples))


def benchmark_full_backward(task, model, optimizer, training, device, warmup, steps, train_step=10):
    """Reference full-loss backward, with fresh forward graphs and frozen weights."""
    samples = []
    for iteration in range(warmup + steps):
        started = []
        def phase(name):
            if name == 'backward':
                synchronize(device)
                if device.type == 'cuda':
                    a, b = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                    a.record()
                else:
                    a = b = None
                started[:] = [time.perf_counter(), a, b]
            elif name == 'optimizer':
                began, a, b = started
                if b is not None:
                    b.record()
                synchronize(device)
                elapsed = time.perf_counter() - began
                if iteration >= warmup:
                    samples.append((elapsed, a.elapsed_time(b)/1000 if a is not None else None))
        take_step(task, model, optimizer, train_step, training, phase_callback=phase)
    return summarize(samples)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--task', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--set', dest='overrides', action='append', default=[])
    parser.add_argument('--prepare-steps', type=int, default=10)
    parser.add_argument('--warmup-steps', type=int, default=5)
    parser.add_argument('--measured-steps', type=int, default=20)
    parser.add_argument('--components', nargs='+', choices=COMPONENTS, default=list(COMPONENTS))
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/backward_latency'))
    parser.add_argument('--upload-wandb', action='store_true')
    args = parser.parse_args(argv)
    if args.prepare_steps < 0 or args.warmup_steps < 0 or args.measured_steps < 1:
        parser.error('Step counts must be nonnegative and measured-steps positive')
    config = load_config(args.config, args.overrides)
    device = torch.device(args.device)
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable')
        torch.cuda.set_device(device.index or 0)
    previous = torch.get_default_device()
    try:
        torch.set_default_device(device)
        np.random.seed(config.training.seed)
        torch.manual_seed(config.training.seed)
        tasks = preprocess_tasks(config.training.split, [args.task],
                                 multitensor_constraints=config.model.multitensor_constraints)
        if len(tasks) != 1:
            raise ValueError('Expected exactly one selected task')
        task = tasks[0]
        model = ARCCompressor(task, config.model)
        optimizer = create_optimizer(model.weights_list, config.training)
        with runtime_device_dispatch(config.training.get('runtime_device_dispatch', 'global')):
            for step in range(args.prepare_steps):
                take_step(task, model, optimizer, step, config.training)
            # Probe and all benchmarks leave weights fixed at the prepared snapshot.
            with patch.object(optimizer, 'step', lambda: None):
                full = benchmark_full_backward(task, model, optimizer, config.training, device,
                                               args.warmup_steps, args.measured_steps, args.prepare_steps)
                with capture_components(model, args.components) as records:
                    take_step(task, model, optimizer, args.prepare_steps, config.training)
                rows = []
                for record in records:
                    row = benchmark_component(record, model.weights_list, device,
                                              args.warmup_steps, args.measured_steps)
                    rows.append(row)
                    print(f"{row['name']}: {row['wall_median_seconds']*1000:.3f} ms median", flush=True)
    finally:
        torch.set_default_device(previous)
    from omegaconf import OmegaConf
    report = dict(task=args.task, device=str(device), torch=str(torch.__version__),
                  gpu=torch.cuda.get_device_name(device) if device.type=='cuda' else None,
                  config=OmegaConf.to_container(config, resolve=True),
                  prepare_training_steps=args.prepare_steps, warmup_steps=args.warmup_steps,
                  measured_steps=args.measured_steps, full_backward=full,
                  components=sorted(rows, key=lambda r:r['wall_median_seconds'], reverse=True),
                  caveats=[
                      'Isolated VJPs use observed task activations and loss upstream gradients at a fixed weight snapshot.',
                      'Input boundaries are detached; parameter aliases and input strides are preserved.',
                      'Fresh forward graph, clears and forward computation are excluded from each timed backward.',
                      'Completion timing adds CUDA barriers; CUDA event spans include CPU dispatch gaps, not GPU active time.',
                      'Components cannot be summed to reconstruct full backward: boundaries, scheduling and accumulation differ.',
                      'Identity outputs can share upstream gradient paths; results are diagnostic, not exact end-to-end attribution.',
                      'No profiler, optimiser updates, numerical diagnostics or W&B logging inside timed intervals.'])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir/'backward_latency.json').write_text(json.dumps(report, indent=2))
    lines = [f"Full backward median: {full['wall_median_seconds']*1000:.3f} ms", '',
             'Isolated component | backward median ms | mean ms']
    lines += [f"{r['name']} | {r['wall_median_seconds']*1000:.3f} | {r['wall_mean_seconds']*1000:.3f}" for r in report['components']]
    lines += ['', *report['caveats']]
    (args.output_dir/'backward_latency.txt').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))
    if args.upload_wandb:
        from profile_layers import upload_profile
        upload_profile(args.output_dir, config.logging.get('wandb_entity') or 'arc_agi',
                       config.logging.wandb_project, 'd4-conv-backward-latency')
    return report


if __name__ == '__main__':
    main()
