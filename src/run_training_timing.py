"""Thin launcher: run the existing train.main with the agreed timing metrics."""
import argparse
import gc
from pathlib import Path

import torch
import train

DEFAULT_TASKS = ["694f12f3", "760b3cac", "94f9d214", "3428a4f5", "dae9d2b5"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="Same model/training YAML as exploratory runs")
    parser.add_argument("--task", action="append", help="Repeat for multiple real ARC task IDs")
    parser.add_argument("--variant", action="append", choices=("primitives", "d4", "optimised"))
    parser.add_argument("--warmup-iterations", type=int, default=10)
    parser.add_argument("--measured-iterations", type=int, default=30)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--clean-only", action="store_true", help="Skip the separate operation-profile pass")
    args = parser.parse_args(argv)
    tasks = args.task or DEFAULT_TASKS
    variants = args.variant or ["primitives", "d4", "optimised"]
    passes = ["clean"] if args.clean_only else ["clean", "operations"]
    for task in tasks:
        for variant in variants:
            for pass_name in passes:
                print(f"{task} / {variant} / {pass_name}", flush=True)
                command = ["--mode", "benchmark", "--task", task,
                           "--benchmark-pass", pass_name,
                           "--warmup-iterations", str(args.warmup_iterations),
                           "--measured-iterations", str(args.measured_iterations),
                           "--set", "logging.wandb=true"]
                if args.config:
                    command.extend(["--config", str(args.config)])
                for override in args.overrides:
                    command.extend(["--set", override])
                command.extend(["--set", f"model.cummax_implementation={variant}"])
                # Both passes enter the same trainer and reseed its normal initializer.
                train.main(command)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
