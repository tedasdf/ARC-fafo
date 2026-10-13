"""Ranking observations preserve decisions and explain raw/EMA promotions."""

import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from compressarc.train.metrics import SolutionTracker
from compressarc.train.ranking_diagnostics import solution_id
from compressarc.train import logging as training_logging


def task():
    return SimpleNamespace(n_train=0, n_test=1, n_colors=1, n_x=1, n_y=1,
                           colors=[0, 2], in_out_same_size=True, all_out_same_size=True,
                           shapes=[((1, 1), (1, 1))], solution_hash=None)


def outputs(winner=1):
    logits = torch.zeros(1, 2, 1, 1, 2)
    logits[0, winner, 0, 0, 1] = 8
    return logits, torch.ones(1, 1, 2), torch.ones(1, 1, 2)


def test_diagnostics_do_not_change_scores_ema_or_guess_order():
    normal, observed = SolutionTracker(task()), SolutionTracker(task())
    observed.enable_diagnostics()
    for index, step in enumerate([0, 1, 149, 150, 1062, 1063, 1499]):
        sample = outputs(index % 2)
        normal.update(step, sample)
        observed.update(step, sample)
        assert observed.scores == normal.scores
        assert observed.solution_most_frequent == normal.solution_most_frequent
        assert observed.solution_second_most_frequent == normal.solution_second_most_frequent
        torch.testing.assert_close(observed.ema_logits, normal.ema_logits, rtol=0, atol=0)
        assert [row["source"] for row in observed.diagnostics.last["candidates"]] == ["raw", "ema"]
        json.dumps(observed.diagnostics.last, allow_nan=False)


def test_same_grid_can_have_both_sources_and_separate_score_mass():
    tracker = SolutionTracker(task())
    tracker.enable_diagnostics()
    tracker.update(150, outputs())
    raw, ema = tracker.diagnostics.last["candidates"]
    selected = tracker.diagnostics.last["selected"][0]
    assert raw["id"] == ema["id"] == selected["id"]
    assert selected["raw_count"] == selected["ema_count"] == 1
    assert raw["accumulated_before"] is None
    assert ema["accumulated_before"] == raw["accumulated_after"]
    assert selected["accumulated_score"] == pytest.approx(np.logaddexp(selected["raw_log_mass"], selected["ema_log_mass"]))
    assert selected["last_source"] == "ema"
    assert selected["last_top1_promotion_source"] == "raw"
    assert raw["contribution_score"] == -10 * raw["uncertainty"]
    assert ema["contribution_score"] == -10 * ema["uncertainty"] - 4


def test_promotion_records_trigger_source_and_previous_guess_moves_to_second():
    tracker = SolutionTracker(task())
    tracker.enable_diagnostics()
    tracker.update(0, outputs(1))
    old_id = tracker.diagnostics.last["selected"][0]["id"]
    tracker.update(150, outputs(0))
    raw = tracker.diagnostics.last["candidates"][0]
    first, second = tracker.diagnostics.last["selected"]
    assert raw["top1_changed"]
    assert raw["top2_after"] == old_id
    assert first["id"] == raw["id"]
    assert second["id"] == old_id
    assert first["last_top1_promotion_source"] == "raw"


def test_grid_identity_stable_and_sensitive_to_shape_and_color():
    assert solution_id((((2, 0),),)) == solution_id([[[2, 0]]])
    assert solution_id([[[2, 0]]]) != solution_id([[[0, 2]]])
    assert solution_id([[[2, 0]]]) != solution_id([[[2], [0]]])


class Table:
    def __init__(self, columns):
        self.columns, self.data = columns, []
    def add_data(self, *values):
        self.data.append(values)


class Artifact:
    def __init__(self, name, type):
        self.name, self.type, self.files = name, type, []
    def add_file(self, path, name):
        self.files.append((path, name))


@pytest.fixture
def fake_wandb(monkeypatch):
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(
        Table=Table, Image=lambda image, **kwargs: "image", Artifact=Artifact,
    ))
    monkeypatch.setattr(training_logging, "plot_predictions", lambda *a: object())
    monkeypatch.setattr(training_logging, "training_predictions_table", lambda *a: "train-table")
    import matplotlib.pyplot as plt
    monkeypatch.setattr(plt, "close", lambda *a: None)


def test_logging_scalars_every_update_and_tables_preserve_all_steps(fake_wandb):
    tracker = SolutionTracker(task())
    tracker.enable_diagnostics()
    records = []
    run = SimpleNamespace(log=records.append)
    metrics = {"loss": 1., "kl": 0., "kl_components": {}, "reconstruction_error": .1, "outputs": outputs()}
    tracker.update(150, outputs())
    training_logging.log_training_step(run, task(), tracker, 150, metrics)
    assert len(tracker.diagnostics.pending) == 1
    assert "ranking/raw_uncertainty" in records[0]
    assert "ranking/candidates" not in records[0]
    tracker.update(151, outputs())
    training_logging.log_training_step(run, task(), tracker, 151, metrics, include_prediction=True)
    assert len(records[1]["ranking/candidates"].data) == 4
    assert len(records[1]["ranking/selected"].data) == 4
    assert {row[0] for row in records[1]["ranking/candidates"].data} == {150, 151}
    assert tracker.diagnostics.pending == []


def test_training_writes_complete_trace_and_uploads_after_close(fake_wandb, monkeypatch, tmp_path):
    import train
    from config import load_config
    from preprocessing import Task
    actual_task = Task("ranking_trace", {
        "train": [{"input": [[0]], "output": [[2]]}],
        "test": [{"input": [[0]]}],
    }, None)
    config = load_config(overrides=["training.iterations=21", "training.device=cpu",
        "logging.wandb=true", "logging.solution_diagnostics=true", "logging.prediction_every=10"])
    records, artifacts = [], []
    run = SimpleNamespace(id="trace-test", summary={}, log=records.append,
                          log_artifact=artifacts.append, finish=lambda: None)
    monkeypatch.setattr(train, "load_config", lambda *a, **k: config)
    monkeypatch.setattr(train.preprocessing, "preprocess_tasks", lambda *a, **k: [actual_task])
    monkeypatch.setattr(train, "ARCCompressor", lambda *a: SimpleNamespace(weights_list=[torch.nn.Parameter(torch.zeros(1))]))
    monkeypatch.setattr(train, "initialize_wandb", lambda *a: run)
    monkeypatch.setattr(train, "log_problem", lambda *a: None)
    monkeypatch.setattr(train, "log_latent_pca", lambda *a: None)
    requested = []
    def step(task, model, optimizer, index, config, return_outputs=False):
        requested.append(return_outputs)
        logits = torch.zeros(2, 2, 1, 1, 2)
        logits[:, 1, :, :, 1] = 8
        return {"loss": 1., "kl": 0., "kl_components": {}, "reconstruction_error": .1,
                "outputs": (logits, torch.ones(2, 1, 2), torch.ones(2, 1, 2))}
    monkeypatch.setattr(train, "take_step", step)
    original_device = torch.get_default_device()
    try:
        train.main(["--task", "ranking_trace", "--output-dir", str(tmp_path)])
    finally:
        torch.set_default_device(original_device)
    directory = tmp_path / "ranking_trace-trace-test.ranking"
    trace = directory / "events.jsonl"
    snapshots = [json.loads(line) for line in trace.read_text().splitlines()]
    assert [row["train_step"] for row in snapshots] == list(range(21))
    assert all(len(row["candidates"]) == len(row["selected"]) == 2 for row in snapshots)
    assert requested == [True] * 21
    assert len(records[-1]["ranking/candidates"].data) == 2
    assert artifacts[0].type == "ranking-diagnostics"
    assert {name for path, name in artifacts[0].files} == {
        "events.jsonl", "grids.jsonl", "metadata.json", "training_predictions.jsonl",
    }
    grid_entries = [json.loads(line) for line in (directory / "grids.jsonl").read_text().splitlines()]
    assert len(grid_entries) == len({entry["id"] for entry in grid_entries}) == 1
    registry = {entry["id"] for entry in grid_entries}
    assert all(candidate["id"] in registry for snapshot in snapshots for candidate in snapshot["candidates"])
    metadata = json.loads((directory / "metadata.json").read_text())
    assert metadata["resolved_config"]["training"]["iterations"] == 21
    assert metadata["seed"] == config.training.seed
    assert "code_commit" in metadata and len(metadata["source_sha256"]) == 64
    predictions = [json.loads(line) for line in (directory / "training_predictions.jsonl").read_text().splitlines()]
    assert [record["train_step"] for record in predictions] == [9, 19, 20]
    assert all(record["examples"][0]["y_true"] == record["examples"][0]["y_pred"] == [[2]] for record in predictions)



def test_events_have_separate_penalties_and_intermediate_rank_scores():
    tracker = SolutionTracker(task())
    tracker.enable_diagnostics()
    tracker.update(149, outputs())
    early_raw, early_ema = tracker.diagnostics.last["candidates"]
    assert early_raw["train_step"] == early_ema["train_step"] == 149
    assert early_raw["early_step_penalty"] == early_ema["early_step_penalty"] == -10
    assert early_raw["ema_penalty"] == 0 and early_ema["ema_penalty"] == -4
    assert [row["contribution_index"] for row in (early_raw, early_ema)] == [0, 1]
    for row in (early_raw, early_ema):
        assert "grid" not in row
        assert len(row["top_two_after"]) == 2
        assert row["top_two_after"][0]["id"] == row["top1_after"]
        assert row["top_two_after"][1]["id"] == row["top2_after"]
        assert row["top_two_after"][0]["accumulated_score"] == row["top1_after_score"]
        assert row["contribution_score"] == pytest.approx(row["confidence_score"] + row["ema_penalty"] + row["early_step_penalty"])
    assert early_raw["top_two_after"][0]["ema_count"] == 0
    assert early_ema["top_two_after"][0]["ema_count"] == 1
    tracker.update(150, outputs())
    assert all(row["early_step_penalty"] == 0 for row in tracker.diagnostics.last["candidates"])
    assert len(tracker.diagnostics.grids) == len(tracker.diagnostics.unwritten_grids) == 1
    assert all("grid" not in row for row in tracker.diagnostics.last["selected"])
    assert early_raw["top_two_after"][0]["ema_count"] == 0


def test_metadata_records_resolved_config_seed_and_actual_commit(monkeypatch):
    from config import load_config
    from compressarc.train import ranking_diagnostics
    def git(command, **kwargs):
        return SimpleNamespace(stdout=" M src/train.py\n" if "status" in command else "123456abcdef\n")
    monkeypatch.setattr(ranking_diagnostics.subprocess, "run", git)
    config = load_config(overrides=["training.seed=42", "training.iterations=17"])
    metadata = ranking_diagnostics.run_metadata(config, "50cb2852", "test-run")
    assert metadata["code_commit"] == "123456abcdef"
    assert metadata["source_dirty"] is True
    assert metadata["seed"] == metadata["resolved_config"]["training"]["seed"] == 42
    assert metadata["resolved_config"]["training"]["iterations"] == 17
    assert metadata["contribution_order"] == ["raw", "ema"]
