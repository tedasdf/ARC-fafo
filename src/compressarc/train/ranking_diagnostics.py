"""Optional observations of candidate scores; never used to choose guesses."""

import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np


def solution_id(solution):
    if solution is None:
        return None
    encoded = json.dumps(solution, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def run_metadata(config, task_name, run_id, config_source=None):
    """Record resolved settings and actual source revision without guessing."""
    from omegaconf import OmegaConf
    import torch
    root = Path(__file__).resolve().parents[3]
    commit, dirty, git_error = None, None, None
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True,
                                capture_output=True, text=True, timeout=10).stdout.strip()
        status = subprocess.run(["git", "status", "--porcelain", "--", "src"], cwd=root,
                                check=True, capture_output=True, text=True, timeout=10)
        dirty = bool(status.stdout.strip())
    except (OSError, subprocess.SubprocessError) as error:
        git_error = type(error).__name__
    fingerprint = hashlib.sha256()
    paths = [root / "src/train.py", root / "src/run_training_timing.py", root / "src/preprocessing.py"]
    paths += list((root / "src/compressarc").rglob("*.py"))
    paths += list((root / "src/config").rglob("*.yaml"))
    for path in sorted(paths):
        fingerprint.update(path.relative_to(root).as_posix().encode() + b"\0" + path.read_bytes() + b"\0")
    return {"schema_version": 2, "task": task_name, "run_id": run_id,
            "resolved_config": OmegaConf.to_container(config, resolve=True),
            "config_source": str(config_source) if config_source else None,
            "seed": int(config.training.seed), "code_commit": commit, "source_dirty": dirty,
            "git_error": git_error, "source_sha256": fingerprint.hexdigest(),
            "python": sys.version, "python_executable": sys.executable,
            "torch_version": torch.__version__, "numpy_version": np.__version__,
            "cuda_version": torch.version.cuda, "step_numbering": "train_step is zero-based; iteration is one-based",
            "contribution_order": ["raw", "ema"]}

class RankingDiagnostics:
    def __init__(self):
        self.sources = {}
        self.grids = {}
        self.unwritten_grids = []
        self.pending_grids = []
        self.last = None
        self.pending = []

    def start_step(self, step):
        self.last = {"train_step": int(step), "iteration": int(step) + 1, "candidates": [], "selected": []}

    def record_candidate(self, tracker, source, grid, uncertainty, score, previous_score, previous_top1, previous_top2, early_step_penalty, ema_penalty):
        identity = solution_id(grid)
        if identity not in self.grids:
            self.grids[identity] = grid
            entry = {"id": identity, "first_train_step": self.last["train_step"], "grid": grid}
            self.unwritten_grids.append(entry)
            self.pending_grids.append(entry)
        counts = self.sources.setdefault(identity, {
            "raw_count": 0, "ema_count": 0, "raw_log_mass": None, "ema_log_mass": None,
            "last_source": None, "last_uncertainty": None, "last_contribution": None,
            "last_top1_promotion_source": None,
            "first_train_step": self.last["train_step"], "last_train_step": self.last["train_step"],
        })
        counts[source + "_count"] += 1
        counts["last_train_step"] = self.last["train_step"]
        mass_key = source + "_log_mass"
        counts[mass_key] = float(np.logaddexp(-np.inf if counts[mass_key] is None else counts[mass_key], score))
        counts.update(last_source=source, last_uncertainty=float(uncertainty), last_contribution=float(score))
        first, second = tracker.solution_most_frequent, tracker.solution_second_most_frequent
        first_changed, second_changed = first != previous_top1, second != previous_top2
        if first_changed and first == grid:
            counts["last_top1_promotion_source"] = source
        self.last["candidates"].append({
            "train_step": self.last["train_step"], "contribution_index": len(self.last["candidates"]),
            "source": source, "id": identity, "uncertainty": float(uncertainty),
            "confidence_score": -10 * float(uncertainty),
            "early_step_penalty": early_step_penalty, "ema_penalty": ema_penalty,
            "contribution_score": float(score),
            "accumulated_before": float(previous_score) if math.isfinite(previous_score) else None,
            "accumulated_after": tracker.scores[hash(grid)],
            "top1_after": solution_id(first), "top2_after": solution_id(second),
            "top1_after_score": tracker.scores[hash(first)], "top2_after_score": tracker.scores[hash(second)],
            "top_two_after": self.selected_snapshot(tracker),
            "top1_changed": first_changed, "top2_changed": second_changed,
        })

    def selected_snapshot(self, tracker):
        result = []
        for rank, grid in enumerate((tracker.solution_most_frequent, tracker.solution_second_most_frequent), 1):
            identity = solution_id(grid)
            result.append({"rank": rank, "id": identity,
                           "accumulated_score": tracker.scores[hash(grid)], **self.sources[identity]})
        return result

    def finish_step(self, tracker):
        self.last["selected"] = self.selected_snapshot(tracker)
        self.pending.append(self.last)

    def candidate_ranking(self, tracker):
        """Rank all unique candidates by accumulated score; ties retain arrival order."""
        ordered = sorted(self.grids, key=lambda identity: tracker.scores[hash(self.grids[identity])], reverse=True)
        return [{"rank": rank, "id": identity,
                 "accumulated_score": tracker.scores[hash(self.grids[identity])],
                 "appearance_count": self.sources[identity]["raw_count"] + self.sources[identity]["ema_count"],
                 **self.sources[identity]} for rank, identity in enumerate(ordered, 1)]

    def scalar_metrics(self):
        payload = {}
        for candidate in self.last["candidates"]:
            prefix = "ranking/" + candidate["source"]
            for field in ("id", "uncertainty", "confidence_score", "early_step_penalty", "ema_penalty", "contribution_score", "accumulated_after", "top1_after", "top2_after", "top1_after_score", "top2_after_score"):
                payload[prefix + "_" + field] = candidate[field]
        for selected in self.last["selected"]:
            prefix = f"ranking/top_{selected['rank']}"
            payload[prefix + "_id"] = selected["id"]
            payload[prefix + "_score"] = selected["accumulated_score"]
        payload["ranking/score_gap"] = self.last["selected"][0]["accumulated_score"] - self.last["selected"][1]["accumulated_score"]
        return payload
