import argparse
import ast
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

import arc_compressor
import layers
import multitensor_systems
import preprocessing
import solution_selection
import train
import visualization


"""
Train one freshly initialized CompressARC model on one ARC task.

Metrics and visualizations can be streamed to Weights & Biases. This script does
not persist run metrics, model state, or plots locally.
"""

np.random.seed(0)
torch.manual_seed(0)
torch.set_default_dtype(torch.float32)
torch.set_default_device('cuda' if torch.cuda.is_available() else 'cpu')


def parse_args():
    parser = argparse.ArgumentParser(description='Train CompressARC on one ARC task.')
    parser.add_argument('--split', choices=('training', 'evaluation', 'test'))
    task_group = parser.add_mutually_exclusive_group()
    task_group.add_argument('--task', dest='task_name', help='One ARC task ID')
    task_group.add_argument(
        '--tasks',
        dest='task_names',
        nargs='+',
        help='Several ARC task IDs to run sequentially as separate W&B runs',
    )
    parser.add_argument('--iterations', type=int, default=1500)
    parser.add_argument('--shift-variant', choices=('original', 'projected_conv', 'pure_conv'), default='original')
    parser.add_argument('--direction-share-variant', choices=('original', 'projected_d4', 'pure_d4'), default='original')
    parser.add_argument('--cummax-variant', choices=('original', 'projected_lse', 'pure_lse'), default='original')
    parser.add_argument('--early-stop', action='store_true')
    parser.add_argument('--early-stop-window', type=int, default=50)
    parser.add_argument('--early-stop-epsilon', type=float, default=1e-4)
    parser.add_argument('--early-stop-warmup', type=int, default=300)
    parser.add_argument('--wandb', action='store_true', help='Log this run to Weights & Biases')
    parser.add_argument('--wandb-project', default='compressarc')
    parser.add_argument('--wandb-entity', default=None)
    parser.add_argument(
        '--wandb-tags',
        nargs='+',
        default=[],
        help='Additional tags to attach to every W&B run',
    )
    parser.add_argument('--wandb-mode', choices=('online', 'offline', 'disabled'), default='online')
    parser.add_argument('--wandb-log-every', type=int, default=1)
    parser.add_argument('--prediction-every', type=int, default=50)
    parser.add_argument(
        '--multitensor-constraints',
        nargs='+',
        choices=('strict', 'relaxed'),
        default=['strict'],
        help='Run one or both legal-dimension policies as separate W&B runs',
    )
    parser.add_argument('--skip-pca', action='store_true', help='Skip final latent PCA analysis')
    parser.add_argument('--pca-samples', type=int, default=100)
    parser.add_argument('--pca-kl-threshold', type=float, default=1.0)
    parser.add_argument('--pca-components', type=int, default=3)
    parser.add_argument('--pca-max-panels', type=int, default=24)
    args = parser.parse_args()

    if args.iterations < 1:
        parser.error('--iterations must be at least 1')
    if args.wandb_log_every < 1:
        parser.error('--wandb-log-every must be at least 1')
    if args.prediction_every < 1:
        parser.error('--prediction-every must be at least 1')
    if args.pca_samples < 1:
        parser.error('--pca-samples must be at least 1')
    if args.pca_components < 1:
        parser.error('--pca-components must be at least 1')
    if args.pca_max_panels < 1:
        parser.error('--pca-max-panels must be at least 1')
    if args.early_stop_window < 1 or args.early_stop_warmup < 1:
        parser.error('early-stop window and warmup must be positive')
    if args.early_stop_epsilon < 0:
        parser.error('--early-stop-epsilon must be nonnegative')
    return args


def kl_metric_name(component_name):
    """Turn a multitensor dimension string into a readable metric name."""
    dims = ast.literal_eval(component_name)
    axis_names = ('example', 'color', 'direction', 'height', 'width')
    active_axes = [name for name, is_active in zip(axis_names, dims) if is_active]
    return 'kl_components/' + '_'.join(active_axes)


def figure_to_rgb(figure):
    """Render a Matplotlib figure to memory without creating an image file."""
    figure.canvas.draw()
    return np.asarray(figure.canvas.buffer_rgba())[..., :3].copy()


def parameter_counts(model):
    """Count unique optimizer tensors so tied references are counted once."""
    parameters = {id(p): p for p in model.weights_list}
    kernels = {id(p): p for conv in getattr(model, 'tied_convs', []) for p in conv.parameters()}
    return {
        'parameter_count': sum(p.numel() for p in parameters.values()),
        'trainable_parameter_count': sum(p.numel() for p in parameters.values() if p.requires_grad),
        'parameter_tensor_count': len(parameters),
        'convolution_parameter_count': sum(p.numel() for p in kernels.values()),
    }


def log_parameters(run, model, train_step):
    """Log a weight distribution and all 3x3 kernel values at snapshot cadence."""
    if run is None:
        return
    import wandb
    parameters = list({id(p): p for p in model.weights_list}.values())
    values = np.concatenate([p.detach().float().cpu().numpy().ravel()
                             for p in parameters if p.numel()])
    metrics = {'train_step': train_step, 'parameters/values': wandb.Histogram(values)}
    run.summary['parameters_with_grad_count'] = sum(p.numel() for p in parameters if p.grad is not None)
    for depth, conv in enumerate(getattr(model, 'tied_convs', [])):
        for name, parameter in conv.named_parameters():
            kernel = parameter.detach().cpu().tolist()
            run.summary[f'kernels/layer_{depth}/{name}'] = kernel
            for row, entries in enumerate(kernel):
                for column, value in enumerate(entries):
                    metrics[f'kernels/layer_{depth}/{name}/{row}_{column}'] = value
    for depth, module in enumerate(getattr(model, 'd4_direction_shares', [])):
        theta = module.theta.detach().cpu().tolist()
        run.summary[f'direction_share/layer_{depth}/theta'] = theta
        for orbit, value in enumerate(theta):
            metrics[f'direction_share/layer_{depth}/theta_{orbit}'] = value
    for depth, module in enumerate(getattr(model, 'lse_modules', [])):
        for name, parameter in module.named_parameters():
            values = parameter.detach().cpu().tolist()
            run.summary[f'lse/layer_{depth}/{name}'] = values
            for index, value in enumerate(values):
                metrics[f'lse/layer_{depth}/{name}_{index}'] = value
    run.log(metrics)


def initialize_wandb(args, task, optimizer, model):
    if not args.wandb:
        return None

    try:
        import wandb
    except ImportError as error:
        raise RuntimeError(
            'W&B tracking requires the wandb package. Run: pip install -r requirements.txt'
        ) from error

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=(f'{task.task_name}-{args.split}-{task.multitensor_constraints}'
              f'-shift_{args.shift_variant}-dir_{args.direction_share_variant}'
              f'-cummax_{args.cummax_variant}'),
        mode=args.wandb_mode,
        config={
            'task_name': task.task_name,
            'split': args.split,
            'shift_variant': args.shift_variant,
            'direction_share_variant': args.direction_share_variant,
            'cummax_variant': args.cummax_variant,
            'early_stop': args.early_stop,
            'early_stop_window': args.early_stop_window,
            'early_stop_epsilon_relative': args.early_stop_epsilon,
            'early_stop_warmup': args.early_stop_warmup,
            'n_train_examples': task.n_train,
            'n_test_examples': task.n_test,
            'n_examples': task.n_examples,
            'n_colors': task.n_colors + 1,
            'canvas_height': task.n_x,
            'canvas_width': task.n_y,
            'iterations': args.iterations,
            'prediction_every': args.prediction_every,
            'pca_enabled': not args.skip_pca,
            'pca_samples': args.pca_samples,
            'pca_kl_threshold': args.pca_kl_threshold,
            'pca_components': args.pca_components,
            'pca_max_panels': args.pca_max_panels,
            'optimizer': type(optimizer).__name__,
            'learning_rate': optimizer.param_groups[0]['lr'],
            'adam_betas': optimizer.param_groups[0]['betas'],
            'decoding_dim': arc_compressor.ARCCompressor.decoding_dim,
            'n_layers': arc_compressor.ARCCompressor.n_layers,
            'device': torch.get_default_device().type,
            'stable_solve_definition': 'first step in final uninterrupted pass@2-correct streak',
            'flops_definition': 'PyTorch profiler supported-op estimate for one full training step',
            'multitensor_constraints': task.multitensor_constraints,
            'legal_multitensor_count': sum(1 for _ in task.multitensor_system),
            **parameter_counts(model),
        },
        tags=[
            'arc-agi',
            args.split,
            task.multitensor_constraints,
            args.shift_variant,
            args.direction_share_variant,
            args.cummax_variant,
            *args.wandb_tags,
        ],
        save_code=True,
    )
    run.define_metric('train_step')
    run.define_metric('train/*', step_metric='train_step')
    run.define_metric('kl_components/*', step_metric='train_step')
    run.define_metric('predictions/*', step_metric='train_step')
    run.define_metric('parameters/*', step_metric='train_step')
    run.define_metric('kernels/*', step_metric='train_step')
    run.define_metric('solve/*', step_metric='train_step')
    run.define_metric('compute/*', step_metric='train_step')
    run.summary.update(parameter_counts(model))
    return run


def log_problem(run, logger):
    if run is None:
        return

    import wandb

    figure = visualization.plot_problem(logger, fname=False)
    run.log({
        'puzzle/problem': wandb.Image(
            figure_to_rgb(figure),
            caption=(
                f'{logger.task.task_name}: {logger.task.n_train} train + '
                f'{logger.task.n_test} test examples'
            ),
        )
    })
    plt.close(figure)


def log_training_step(run, logger, train_step, include_prediction=False):
    if run is None:
        return

    metrics = {
        'train_step': train_step,
        'train/loss': logger.loss_curve[-1],
        'train/reconstruction_error': logger.reconstruction_error_curve[-1],
        'train/total_KL': logger.total_KL_curve[-1],
    }
    metrics.update({
        kl_metric_name(component_name): curve[-1]
        for component_name, curve in logger.KL_curves.items()
    })

    prediction_figure = None
    if include_prediction:
        import wandb

        prediction_figure = visualization.plot_solution(logger, fname=False)
        metrics['predictions/solutions'] = wandb.Image(
            figure_to_rgb(prediction_figure),
            caption=f'Solutions after step {train_step + 1}',
        )

    run.log(metrics)
    if prediction_figure is not None:
        plt.close(prediction_figure)


def current_correctness(task, logger):
    """Return top-1 and pass@2 correctness for the current ranked guesses."""
    if task.solution_hash is None:
        return None, None
    top_1 = hash(logger.solution_most_frequent) == task.solution_hash
    guess_2 = hash(logger.solution_second_most_frequent) == task.solution_hash
    return top_1, top_1 or guess_2


def estimate_training_step_flops(step_callable):
    """Run one real training step and return profiler-counted FLOPs."""
    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.profiler.profile(activities=activities, with_flops=True) as profile:
        step_callable()
    return int(sum(event.flops for event in profile.key_averages()))


def solve_metrics(correctness_history, elapsed_history, flops_per_step):
    """Compute first solve and start of the final uninterrupted solve streak."""
    solved_steps = [index for index, correct in enumerate(correctness_history) if correct]
    first_step = solved_steps[0] if solved_steps else None
    stable_step = None
    if correctness_history and correctness_history[-1]:
        stable_step = len(correctness_history) - 1
        while stable_step > 0 and correctness_history[stable_step - 1]:
            stable_step -= 1
    return {
        'first_solve_step': first_step,
        'stable_solve_step': stable_step,
        'wall_clock_to_first_solve_seconds': elapsed_history[first_step] if first_step is not None else None,
        'wall_clock_to_stable_solve_seconds': elapsed_history[stable_step] if stable_step is not None else None,
        'estimated_flops_per_training_step': flops_per_step,
        'estimated_flops_to_first_solve': flops_per_step * (first_step + 1) if first_step is not None else None,
        'estimated_flops_to_stable_solve': flops_per_step * (stable_step + 1) if stable_step is not None else None,
    }


def plot_pca_component(component, axis_names, component_number, strength, max_panels):
    """Visualize one PCA component across a latent tensor's semantic axes."""
    scale = np.max(np.abs(component))
    normalized = component / scale if scale > 0 else component

    if normalized.ndim == 1:
        figure, axis = plt.subplots(figsize=(max(4, normalized.shape[0] * 0.35), 2.5))
        axis.imshow(normalized[None, :], cmap='gray', vmin=-1, vmax=1, aspect='auto')
        axis.set_yticks([])
        axis.set_xlabel(axis_names[0])
    elif normalized.ndim == 2:
        figure, axis = plt.subplots(figsize=(6, 5))
        axis.imshow(normalized, cmap='gray', vmin=-1, vmax=1, aspect='auto')
        axis.set_ylabel(axis_names[0])
        axis.set_xlabel(axis_names[1])
    else:
        leading_shape = normalized.shape[:-2]
        panel_count = min(int(np.prod(leading_shape)), max_panels)
        column_count = min(4, panel_count)
        row_count = int(np.ceil(panel_count / column_count))
        figure, subplot_axes = plt.subplots(
            row_count,
            column_count,
            figsize=(4 * column_count, 3.5 * row_count),
            squeeze=False,
        )
        panels = normalized.reshape((-1,) + normalized.shape[-2:])
        for panel_number, axis in enumerate(subplot_axes.flat):
            if panel_number >= panel_count:
                axis.axis('off')
                continue
            axis.imshow(panels[panel_number], cmap='gray', vmin=-1, vmax=1, aspect='auto')
            leading_index = np.unravel_index(panel_number, leading_shape)
            axis.set_title(', '.join(
                f'{name}={index}' for name, index in zip(axis_names[:-2], leading_index)
            ))
            axis.set_ylabel(axis_names[-2])
            axis.set_xlabel(axis_names[-1])
        if int(np.prod(leading_shape)) > max_panels:
            figure.text(
                0.5,
                0.01,
                f'Showing the first {max_panels} of {int(np.prod(leading_shape))} panels',
                ha='center',
            )

    figure.suptitle(f'Component {component_number + 1}; strength={strength:.5g}')
    figure.tight_layout()
    return figure


def log_latent_principal_components(run, model, args):
    """Sample, average, de-noise, and visualize informative latent multitensors."""
    if run is None or args.skip_pca:
        return

    import wandb

    print(f'Computing latent PCA from {args.pca_samples} decoder samples...')

    @multitensor_systems.multify
    def to_cpu(dims, value):
        return value.detach().cpu()

    @multitensor_systems.multify
    def add_samples(dims, left, right):
        return left + right

    @multitensor_systems.multify
    def finish_average(dims, total):
        mean = total / args.pca_samples
        semantic_axes = tuple(range(mean.ndim - 1))
        return mean - torch.mean(mean, dim=semantic_axes)

    sample_total = None
    with torch.no_grad():
        for _ in tqdm(range(args.pca_samples), desc='latent PCA samples'):
            sample, kl_amounts, kl_names = layers.decode_latents(
                model.target_capacities,
                model.decode_weights,
                model.multiposteriors,
            )
            cpu_sample = to_cpu(sample)
            sample_total = cpu_sample if sample_total is None else add_samples(
                sample_total, cpu_sample
            )
    means = finish_average(sample_total)

    tensor_entries = []
    for kl_amount, kl_name in zip(kl_amounts, kl_names):
        kl_value = float(torch.sum(kl_amount).detach().cpu())
        if kl_value >= args.pca_kl_threshold:
            tensor_entries.append((kl_value, tuple(ast.literal_eval(kl_name))))
    tensor_entries.sort(reverse=True)

    table = wandb.Table(columns=[
        'tensor_dims',
        'semantic_axes',
        'tensor_shape',
        'KL',
        'component',
        'strength',
        'component_heatmap',
    ])
    all_axis_names = ('example', 'color', 'direction', 'height', 'width')

    for kl_value, dims in tensor_entries:
        tensor = means[dims].numpy()
        semantic_shape = tensor.shape[:-1]
        flattened = tensor.reshape(-1, tensor.shape[-1])
        left_vectors, singular_values, _ = np.linalg.svd(flattened, full_matrices=False)
        component_count = min(args.pca_components, len(singular_values))
        active_axis_names = [
            name for name, is_active in zip(all_axis_names, dims) if is_active
        ]

        for component_number in range(component_count):
            component = left_vectors[:, component_number].reshape(semantic_shape)
            strength = float(singular_values[component_number] / flattened.shape[0])
            figure = plot_pca_component(
                component,
                active_axis_names,
                component_number,
                strength,
                args.pca_max_panels,
            )
            table.add_data(
                str(dims),
                ', '.join(active_axis_names),
                str(tensor.shape),
                kl_value,
                component_number + 1,
                strength,
                wandb.Image(figure_to_rgb(figure)),
            )
            plt.close(figure)

    run.log({'latent_analysis/principal_components': table})
    run.summary['pca_significant_tensor_count'] = len(tensor_entries)
    run.summary['pca_kl_threshold'] = args.pca_kl_threshold


def record_final_result(run, task, logger, final_step_had_prediction):
    if run is None:
        return

    if not final_step_had_prediction:
        import wandb

        final_step = len(logger.loss_curve) - 1
        figure = visualization.plot_solution(logger, fname=False)
        run.log({
            'train_step': final_step,
            'predictions/solutions': wandb.Image(
                figure_to_rgb(figure),
                caption=f'Final solutions after step {final_step + 1}',
            ),
        })
        plt.close(figure)

    run.summary['guess_1'] = logger.solution_most_frequent
    run.summary['guess_2'] = logger.solution_second_most_frequent
    if task.solution_hash is not None:
        guess_1_correct, pass_2_correct = current_correctness(task, logger)
        run.summary['top_1_correct'] = guess_1_correct
        run.summary['pass_2_correct'] = pass_2_correct


def run_task(args, split, task_name, multitensor_constraints):
    # Reset per condition so strict/relaxed comparisons do not inherit RNG
    # state from whichever task happened to run first.
    np.random.seed(0)
    torch.manual_seed(0)

    print(
        f'Performing a training run on task {task_name} '
        f'with {multitensor_constraints} multitensor constraints.'
    )
    task = preprocessing.preprocess_tasks(
        split,
        [task_name],
        multitensor_constraints=multitensor_constraints,
    )[0]
    print(
        f'Using {task.n_train} training examples and {task.n_test} test examples '
        f'({task.n_examples} total).'
    )

    model = arc_compressor.ARCCompressor(
        task,
        shift_variant=args.shift_variant,
        direction_share_variant=args.direction_share_variant,
        cummax_variant=args.cummax_variant,
    )
    optimizer = torch.optim.Adam(model.weights_list, lr=0.01, betas=(0.5, 0.9))
    logger = solution_selection.Logger(task)
    wandb_run = initialize_wandb(args, task, optimizer, model)
    log_parameters(wandb_run, model, -1)
    log_problem(wandb_run, logger)

    final_step_had_prediction = False
    correctness_history = []
    elapsed_history = []
    flops_per_step = None
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    training_started = time.perf_counter()
    best_loss = float('inf')
    steps_without_relative_improvement = 0
    stop_reason = 'max_iterations'
    for train_step in tqdm(range(args.iterations)):
        if train_step == 0:
            flops_per_step = estimate_training_step_flops(
                lambda: train.take_step(task, model, optimizer, train_step, logger)
            )
        else:
            train.take_step(task, model, optimizer, train_step, logger)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - training_started
        top_1_correct, pass_2_correct = current_correctness(task, logger)
        correctness_history.append(bool(pass_2_correct))
        elapsed_history.append(elapsed)
        current_loss = logger.loss_curve[-1]
        relative_improvement = (
            (best_loss - current_loss) / max(abs(best_loss), 1e-12)
            if np.isfinite(best_loss) else float('inf')
        )
        if relative_improvement > args.early_stop_epsilon:
            best_loss = current_loss
            steps_without_relative_improvement = 0
        elif train_step + 1 >= args.early_stop_warmup:
            steps_without_relative_improvement += 1
        include_prediction = (train_step + 1) % args.prediction_every == 0
        should_log = train_step % args.wandb_log_every == 0 or include_prediction
        if should_log:
            log_training_step(wandb_run, logger, train_step, include_prediction)
        if include_prediction or train_step == args.iterations - 1:
            log_parameters(wandb_run, model, train_step)
        if wandb_run is not None:
            wandb_run.log({
                'train_step': train_step,
                'solve/top_1_correct': int(bool(top_1_correct)),
                'solve/pass_2_correct': int(bool(pass_2_correct)),
                'solve/elapsed_seconds': elapsed,
                'compute/estimated_cumulative_flops': flops_per_step * (train_step + 1),
            })
        final_step_had_prediction = include_prediction
        if (args.early_stop
                and train_step + 1 >= args.early_stop_warmup
                and steps_without_relative_improvement >= args.early_stop_window):
            stop_reason = 'loss_plateau'
            print(
                f'Early stopping at step {train_step}: no relative loss improvement '
                f'> {args.early_stop_epsilon} for {args.early_stop_window} steps.'
            )
            break

    recorded_solve_metrics = solve_metrics(
        correctness_history, elapsed_history, flops_per_step
    )
    if wandb_run is not None:
        wandb_run.summary.update(recorded_solve_metrics)
        wandb_run.summary.update({
            'iterations_completed': len(logger.loss_curve),
            'early_stop_reason': stop_reason,
            'early_stop_best_loss': best_loss,
            'early_stop_steps_without_improvement': steps_without_relative_improvement,
        })
    record_final_result(wandb_run, task, logger, final_step_had_prediction)
    log_latent_principal_components(wandb_run, model, args)
    if wandb_run is not None:
        wandb_run.finish()

    print('Guess 1:', logger.solution_most_frequent)
    print('Guess 2:', logger.solution_second_most_frequent)
    print('Solve metrics:', recorded_solve_metrics)
    print(f'Finished task {task_name} ({multitensor_constraints}).')

    del logger, optimizer, model, task
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    args = parse_args()
    split = args.split or input(
        'Enter which split you want to find the task in (training, evaluation, test): '
    )
    args.split = split

    if args.task_names:
        task_names = args.task_names
    elif args.task_name:
        task_names = [args.task_name]
    else:
        task_names = [input('Enter which task you want to analyze (eg. 272f95fa): ')]

    conditions = [
        (task_name, constraint_policy)
        for task_name in task_names
        for constraint_policy in args.multitensor_constraints
    ]
    print(
        f'Running {len(conditions)} condition(s) across {len(task_names)} task(s): '
        f'{", ".join(task_names)}'
    )
    for condition_number, (task_name, constraint_policy) in enumerate(conditions, start=1):
        print(
            f'\n[{condition_number}/{len(conditions)}] '
            f'{task_name} ({constraint_policy})'
        )
        run_task(args, split, task_name, constraint_policy)
    print('done')


if __name__ == '__main__':
    main()
