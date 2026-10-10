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
def test_prediction_correctness_history(monkeypatch, correct_guess, expected, include_prediction):
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
        )
    assert [record["train_step"] for record in records] == [49, 99]
    for record in records:
        assert record["train/loss"] == 3.0
        if include_prediction and expected is not None:
            assert record["predictions/top_1_correct"] == expected[0]
            assert record["predictions/pass_2_correct"] == expected[1]
        else:
            # No stale correctness samples between evaluations or without labels.
            assert "predictions/top_1_correct" not in record
            assert "predictions/pass_2_correct" not in record
        assert ("predictions/solutions" in record) == include_prediction
    assert len(figures) == (2 if include_prediction else 0)
