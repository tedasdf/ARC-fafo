"""Check shared trainer benchmark passes, summary logging, and memory boundaries."""
import pytest
import torch

import train
from preprocessing import Task
from compressarc.train.timing import TrainingTiming
from training_metrics_spec import COUNTS, OPERATIONS, SETUP, TRAINING


class RecordingRun:
    def __init__(self):
        self.config = {}
        self.summary = {}
        self.finished = False
        self.history = []
        self.definitions = []
        self.log_step_counts = []
        self.completed_steps = lambda: 0

    def log(self, record):
        assert not self.finished
        self.history.append(dict(record))
        self.log_step_counts.append(self.completed_steps())

    def define_metric(self, name, **kwargs):
        self.definitions.append((name, kwargs))

    def finish(self):
        self.finished = True


@pytest.mark.parametrize("variant", ["primitives", "d4", "optimised"])
def test_clean_and_profile_passes_share_training_and_log_distinct_metrics(monkeypatch, variant):
    problem = {
        "train": [{"input": [[0, 1, 0], [1, 0, 1]], "output": [[0, 1, 0], [1, 0, 1]]}],
        "test": [{"input": [[1, 0, 1], [0, 1, 0]]}],
    }
    task = Task("timing_test", problem, None)
    monkeypatch.setattr(train.preprocessing, "preprocess_tasks", lambda *a, **k: [task])
    models, runs, steps = [], [], []
    original_model = train.ARCCompressor
    original_step = train.take_step

    def capture_model(*args):
        model = original_model(*args)
        models.append(model)
        return model

    def capture_run(*args):
        run = RecordingRun()
        run.completed_steps = lambda: len(steps)
        runs.append(run)
        return run

    def capture_step(*args, **kwargs):
        steps.append(args[3])
        return original_step(*args, **kwargs)

    monkeypatch.setattr(train, "ARCCompressor", capture_model)
    monkeypatch.setattr(train, "initialize_wandb", capture_run)
    monkeypatch.setattr(train, "take_step", capture_step)
    overrides = [
        "training.device=cpu", "model.n_layers=1", "model.channel_dim_shared=2",
        "model.channel_dim_grid=2", "model.share_up_dim=2", "model.share_down_dim=2",
        "model.decoding_dim=2", "model.softmax_dim=2", "model.cummax_dim=2",
        "model.shift_dim=2", "model.nonlinear_dim=2", f"model.cummax_implementation={variant}",
    ]
    base = ["--mode", "benchmark", "--task", "timing_test",
            "--warmup-iterations", "1", "--measured-iterations", "2"]
    for override in overrides:
        base.extend(["--set", override])
    original_device = torch.get_default_device()
    try:
        clean = train.main([*base, "--benchmark-pass", "clean"])
        profiled = train.main([*base, "--benchmark-pass", "operations"])
    finally:
        torch.set_default_device(original_device)
    assert steps == [0, 1, 2, 0, 1, 2]
    assert set(clean) == set(SETUP) | set(TRAINING) | set(COUNTS)
    assert set(profiled) == set(OPERATIONS) | set(COUNTS)
    assert clean["timing/mean_step_ms"] == clean["timing/block_ms"] / 2
    assert clean["timing/warmup_iterations"] == profiled["timing/warmup_iterations"] == 1
    assert clean["timing/measured_iterations"] == profiled["timing/measured_iterations"] == 2
    assert all(profiled[name] >= 0 for name in OPERATIONS)
    assert len(models[0].weights_list) == len(models[1].weights_list)
    for clean_weight, profiled_weight in zip(models[0].weights_list, models[1].weights_list):
        torch.testing.assert_close(clean_weight, profiled_weight, rtol=0, atol=0)
    assert runs[0].summary == clean and runs[1].summary == profiled
    assert runs[0].history == [{"train_step": 0, "timing/first_step_ms": clean["timing/first_step_ms"]}]
    assert runs[0].log_step_counts == [3]  # First-step record also flushes after measurement.
    assert [record["train_step"] for record in runs[1].history] == [1, 2]
    assert runs[1].log_step_counts == [6, 6]  # Flush after every training step has finished.
    assert ("timing/*", {"step_metric": "train_step"}) in runs[1].definitions
    for record in runs[1].history:
        assert set(record) == {"train_step", *OPERATIONS}
        assert all(record[key] >= 0 for key in OPERATIONS)
    for key in OPERATIONS:
        mean = sum(record[key] for record in runs[1].history) / 2
        assert runs[1].summary[key] == pytest.approx(mean)
    assert all(run.finished for run in runs)
    assert [run.config["benchmark_pass"] for run in runs] == ["clean", "operations"]
    assert all(run.config["actual_training_iterations"] == 3 for run in runs)


def test_clean_memory_peaks_reset_after_warmup_before_measured_updates(monkeypatch):
    events = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: events.append("sync"))
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda device: events.append("reset"))
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device: 4 * 2**20)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda device: 6 * 2**20)
    timer = TrainingTiming(torch.device("cuda:0"), "clean", warmup=1, measured=2)

    def step(task, model, optimizer, index, config):
        events.append(f"step{index}")
        return {"loss": 1.0}

    metrics = timer.train(None, None, None, None, step)
    assert events.count("reset") == 1
    reset = events.index("reset")
    assert events.index("step0") < reset < events.index("step1") < events.index("step2")
    assert events[reset - 1] == "sync"
    assert events[reset + 1] == "step1"
    assert metrics["memory/peak_allocated_mib"] == 4
    assert metrics["memory/peak_reserved_mib"] == 6
