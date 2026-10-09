"""Thin launcher: run the existing train.main with the agreed timing metrics."""
import argparse
import gc
from pathlib import Path

import torch
import train

MODEL_CONFIG_DIR = Path(__file__).resolve().parent / "config" / "models"
DEFAULT_CONFIGS = [MODEL_CONFIG_DIR / filename for filename in (
    "original.yaml", "projected_shift.yaml", "projected_direction_share.yaml",
    "projected_cummax_lse.yaml", "all_projected.yaml",
)]

DEFAULT_TASKS = ["694f12f3", "760b3cac", "94f9d214", "3428a4f5", "dae9d2b5"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, action="append",
                        help="Model YAML; repeat to run several. Defaults to all five model configs.")
    parser.add_argument("--task", action="append", help="Repeat for multiple real ARC task IDs")
    parser.add_argument("--warmup-iterations", type=int, default=10)
    parser.add_argument("--measured-iterations", type=int, default=30)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--clean-only", action="store_true", help="Skip the separate operation-profile pass")
    args = parser.parse_args(argv)
    tasks = args.task or DEFAULT_TASKS
    configs = args.config or DEFAULT_CONFIGS
    for config_path in configs:
        if not config_path.is_file():
            parser.error(f"Configuration does not exist: {config_path}")
    passes = ["clean"] if args.clean_only else ["clean", "operations"]
    for task in tasks:
        for config_path in configs:
            for pass_name in passes:
                print(f"{task} / {config_path.stem} / {pass_name}", flush=True)
                command = ["--mode", "benchmark", "--task", task,
                           "--benchmark-pass", pass_name,
                           "--warmup-iterations", str(args.warmup_iterations),
                           "--measured-iterations", str(args.measured_iterations),
                           "--set", "logging.wandb=true"]
                command.extend(["--config", str(config_path)])
                for override in args.overrides:
                    command.extend(["--set", override])
                # Both passes enter the same trainer and reseed its normal initializer.
                train.main(command)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
