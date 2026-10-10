"""Correctness curves use the freshly evaluated guesses and available labels."""

import sys
from types import SimpleNamespace

import pytest

from compressarc.train import logging as training_logging


@pytest.mark.parametrize("correct_guess,expected", [
    ("first", (1, 1)),
    ("second", (0, 1)),
    ("neither", (0, 0)),
    (None, None),
])
@pytest.mark.parametrize("include_prediction", [False, True])
@pytest.mark.parametrize("include_accuracy", [False, True])
def test_prediction_correctness_history(monkeypatch, correct_guess, expected, include_prediction, include_accuracy):
    first, second, other = (((1, 2),),), (((2, 1),),), (((0, 0),),)
    solution = {"first": first, "second": second, "neither": other}.get(correct_guess)
    task = SimpleNamespace(solution_hash=None if solution is None else hash(solution))
    tracker = SimpleNamespace(solution_most_frequent=first, solution_second_most_frequent=second)
    records = []
    run = SimpleNamespace(log=records.append)
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(Image=lambda figure: "prediction-image"))
    figures = []
    monkeypatch.setattr(training_logging, "plot_predictions", lambda *args: figures.append(args) or object())
    import matplotlib.pyplot as plt
    monkeypatch.setattr(plt, "close", lambda figure: None)
    metrics = {"loss": 3.0, "reconstruction_error": 2.0, "kl": 1.0, "kl_components": {}}

    # Successive prediction evaluations remain separate records on train_step.
    for step in (49, 99):
        training_logging.log_training_step(
            run, task, tracker, step, metrics, include_prediction=include_prediction,
            include_accuracy=include_accuracy,
        )
    assert [record["train_step"] for record in records] == [49, 99]
    for record in records:
        assert record["train/loss"] == 3.0
        if (include_accuracy or include_prediction) and expected is not None:
            assert record["predictions/top_1_correct"] == expected[0]
            assert record["predictions/pass_2_correct"] == expected[1]
        else:
            # No stale correctness samples between evaluations or without labels.
            assert "predictions/top_1_correct" not in record
            assert "predictions/pass_2_correct" not in record
        assert ("predictions/solutions" in record) == include_prediction
    assert len(figures) == (2 if include_prediction else 0)

def test_trainer_evaluates_correctness_every_iteration_independently_of_images(monkeypatch):
    import torch
    import train
    from config import load_config
    from preprocessing import Task

    config = load_config(overrides=[
        "training.iterations=4", "training.device=cpu",
        "logging.prediction_every=3", "logging.wandb_log_every=100",
    ])
    task = Task("accuracy_cadence", {
        "train": [{"input": [[0]], "output": [[0]]}],
        "test": [{"input": [[0]]}],
    }, None)
    correct, wrong = (((0,),),), (((1,),),)
    task.solution_hash = hash(correct)
    records, updates, output_requests = [], [], []
    run = SimpleNamespace(log=records.append, finish=lambda: None)
    tracker = SimpleNamespace(solution_most_frequent=None, solution_second_most_frequent=None)

    def update(step, outputs):
        assert outputs == (step,)
        updates.append(step)
        tracker.solution_most_frequent = correct if step % 2 else wrong
        tracker.solution_second_most_frequent = wrong if step % 2 else correct

    tracker.update = update

    def step(task, model, optimizer, index, config, return_outputs=False):
        output_requests.append(return_outputs)
        return {"loss": 3.0, "reconstruction_error": 2.0, "kl": 1.0,
                "kl_components": {}, "outputs": (index,)}

    monkeypatch.setattr(train.preprocessing, "preprocess_tasks", lambda *a, **k: [task])
    monkeypatch.setattr(train, "load_config", lambda *a, **k: config)
    monkeypatch.setattr(train, "ARCCompressor", lambda *a: SimpleNamespace(
        weights_list=[torch.nn.Parameter(torch.zeros(1))],
    ))
    monkeypatch.setattr(train, "SolutionTracker", lambda *a: tracker)
    monkeypatch.setattr(train, "initialize_wandb", lambda *a: run)
    monkeypatch.setattr(train, "take_step", step)
    for name in ("log_problem", "log_final_results", "log_latent_pca"):
        monkeypatch.setattr(train, name, lambda *a, **k: None)
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(Image=lambda figure: "image"))
    monkeypatch.setattr(training_logging, "plot_predictions", lambda *a: object())
    import matplotlib.pyplot as plt
    monkeypatch.setattr(plt, "close", lambda figure: None)
    original_device = torch.get_default_device()
    try:
        train.main(["--task", "accuracy_cadence"])
    finally:
        torch.set_default_device(original_device)

    assert updates == [0, 1, 2, 3]
    assert output_requests == [True] * 4
    assert [record["train_step"] for record in records] == [0, 1, 2, 3]
    assert [record["predictions/top_1_correct"] for record in records] == [0, 1, 0, 1]
    assert [record["predictions/pass_2_correct"] for record in records] == [1] * 4
    assert [record["train_step"] for record in records if "predictions/solutions" in record] == [2, 3]
