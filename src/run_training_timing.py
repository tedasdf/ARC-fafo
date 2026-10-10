"""Shared launcher: call train.main for full training or timing runs."""
import argparse
import gc
import shlex
from pathlib import Path

import torch
import train
from config import load_config

MODEL_CONFIG_DIR = Path(__file__).resolve().parent / "config" / "models"
DEFAULT_CONFIGS = [MODEL_CONFIG_DIR / filename for filename in (
    "original.yaml", "projected_shift.yaml", "projected_direction_share.yaml",
    "projected_cummax_lse.yaml", "all_projected.yaml",
)]

DEFAULT_TASKS = ["694f12f3", "760b3cac", "94f9d214", "3428a4f5", "dae9d2b5"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("train", "benchmark"), default="benchmark")
    parser.add_argument("--dry-run", action="store_true", help="Print planned calls without executing training")
    parser.add_argument("--config", type=Path, action="append",
                        help="Model YAML; repeat to run several. Defaults to all five model configs.")
    parser.add_argument("--task", action="append", help="Repeat for multiple real ARC task IDs")
    parser.add_argument("--warmup-iterations", type=int, default=10)
    parser.add_argument("--measured-iterations", type=int, default=30)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--clean-only", action="store_true", help="Skip the separate operation-profile pass")
    parser.add_argument("--save-checkpoints", action="store_true",
                        help="Save final .pt weights and upload them to W&B in train mode")
    parser.add_argument("--output-dir", type=Path,
                        help="Checkpoint directory; defaults to outputs/training in the trainer")
    args = parser.parse_args(argv)
    if args.save_checkpoints and args.mode != "train":
        parser.error("--save-checkpoints requires --mode train")
    configs = args.config or DEFAULT_CONFIGS
    config_tasks = {}
    for config_path in configs:
        if not config_path.is_file():
            parser.error(f"Configuration does not exist: {config_path}")
        config = load_config(config_path, args.overrides)
        selected = args.task or config.get("experiment", {}).get("task_ids", DEFAULT_TASKS)
        if isinstance(selected, str) or not selected:
            parser.error("experiment.task_ids must be a nonempty list")
        config_tasks[config_path] = [str(task) for task in selected]
    # Preserve explicit duplicate --task flags for cold/warm repeat experiments.
    tasks = args.task or list(dict.fromkeys(
        task for config_path in configs for task in config_tasks[config_path]
    ))
    passes = ["train"] if args.mode == "train" else (
        ["clean"] if args.clean_only else ["clean", "operations"]
    )
    for task in tasks:
        for config_path in configs:
            if task not in config_tasks[config_path]:
                continue
            for pass_name in passes:
                print(f"{task} / {config_path.stem} / {pass_name}", flush=True)
                command = ["--mode", args.mode, "--task", task,
                           "--set", "logging.wandb=true"]
                if args.mode == "benchmark":
                    command.extend(["--benchmark-pass", pass_name,
                                    "--warmup-iterations", str(args.warmup_iterations),
                                    "--measured-iterations", str(args.measured_iterations)])
                command.extend(["--config", str(config_path)])
                if args.save_checkpoints:
                    command.append("--save-checkpoints")
                if args.output_dir is not None:
                    command.extend(["--output-dir", str(args.output_dir)])
                for override in args.overrides:
                    command.extend(["--set", override])
                if args.dry_run:
                    print("python " + shlex.join([str(Path(train.__file__)), *command]))
                    continue
                # Every task enters the same trainer with its configured seed.
                train.main(command)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
