"""Run clean latency and detailed layer profiling as separate training passes."""
import argparse
import json
import subprocess
import sys
from pathlib import Path


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True, type=Path)
    p.add_argument('--task', required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--set', dest='overrides', action='append', default=[])
    p.add_argument('--clean-steps', type=int, default=40)
    p.add_argument('--warmup-steps', type=int, default=10)
    p.add_argument('--capture-steps', type=int, default=1)
    p.add_argument('--no-shapes', action='store_true')
    p.add_argument('--profile-memory', action='store_true')
    p.add_argument('--with-stack', action='store_true')
    p.add_argument('--output-dir', type=Path, default=Path('outputs/d4_conv_layer_profile'))
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args(argv)
    if args.clean_steps <= args.warmup_steps or args.warmup_steps < 1 or args.capture_steps < 1:
        p.error('clean-steps must exceed positive warmup-steps; capture-steps must be positive')
    repo = Path(__file__).resolve().parents[1]
    root = args.output_dir.resolve()
    for capture in (False, True):
        name = 'profile' if capture else 'clean'
        iterations = args.warmup_steps + args.capture_steps if capture else args.clean_steps
        cmd = [sys.executable, str(repo/'src/train.py'), '--config', str(args.config.resolve()),
               '--task', args.task, '--output-dir', str(root/name)]
        overrides = [*args.overrides, f'training.device={args.device}',
                     f'training.iterations={iterations}', 'training.step_timing=phases',
                     f'training.operator_profile={str(capture).lower()}',
                     f'training.operator_profile_start={args.warmup_steps}',
                     f'training.operator_profile_steps={args.capture_steps}',
                     f'training.operator_profile_shapes={str(not args.no_shapes).lower()}',
                     f'training.operator_profile_memory={str(args.profile_memory).lower()}',
                     f'training.operator_profile_stack={str(args.with_stack).lower()}',
                     f'experiment.name=layer_profile_{name}']
        for override in overrides:
            cmd.extend(['--set', override])
        print(subprocess.list2cmdline(cmd), flush=True)
        if not args.dry_run:
            subprocess.run(cmd, cwd=repo, check=True)
    if args.dry_run:
        return
    rows = [json.loads(line) for line in (root/'clean'/f'{args.task}.step_timing.jsonl').read_text().splitlines()]
    clean = rows[args.warmup_steps:]
    keys = sorted({k for r in clean for k in r if k.startswith('timing/')})
    means = {k:sum(r.get(k, 0) for r in clean)/len(clean) for k in keys}
    prefix = root/'profile'/f'{args.task}.operators'
    analysis = json.loads(Path(str(prefix)+'.layers.json').read_text())
    import statistics
    latencies = [r['timing/step_seconds'] for r in clean]
    report = dict(task=args.task, warmup_steps=args.warmup_steps,
                  measured_clean_steps=len(clean), clean_mean_seconds=statistics.mean(latencies),
                  clean_median_seconds=statistics.median(latencies), clean_phase_means=means,
                  profile=analysis,
                  optimization_options=[
                      'Batch existing direction-sharing affine maps if measured launch counts dominate; preserve tied weights and coefficients.',
                      'Fuse pointwise/normalization operations or evaluate compilation if dispatch gaps dominate.',
                      'Evaluate compatible optimizer modes if scalar-read/synchronization counts dominate optimizer host time.',
                      'Consider a custom D4-conv kernel only if shift GPU execution or launch count is substantial.',
                      'Preserve math and validate outputs/gradients, then repeat clean timing on the same task and GPU.'])
    (root/'comparison.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    phase_keys = ['forward','loss','backward','optimizer','metrics','candidates','logging']
    text = [f"Clean full-step mean: {report['clean_mean_seconds']:.6f} s",
            f"Clean full-step median: {report['clean_median_seconds']:.6f} s",
            f"Measured steps: {len(clean)} after {args.warmup_steps} warmup steps", '',
            'Clean phase | host seconds | CUDA-stream seconds (includes dispatch gaps)']
    for name in phase_keys:
        text.append(f"{name} | {means.get('timing/host_'+name+'_seconds',0):.6f} | {means.get('timing/stream_'+name+'_seconds',0):.6f}")
    text += ['', Path(str(prefix)+'.layers.txt').read_text(), '', 'Options to test, selected using the measured bottleneck:', *report['optimization_options']]
    (root/'comparison.txt').write_text('\n'.join(text)+'\n', encoding='utf-8')
    print('\n'.join(text), flush=True)


if __name__ == '__main__':
    main()
