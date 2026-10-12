"""Verify initial reference outputs and training prediction color/crop decoding."""

import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from preprocessing import Task
from compressarc.train import logging as training_logging
from compressarc.train.metrics import SolutionTracker


class RecordingTable:
    def __init__(self, columns):
        self.columns = columns
        self.data = []

    def add_data(self, *values):
        self.data.append(values)


@pytest.fixture
def fake_wandb(monkeypatch):
    fake = SimpleNamespace(Table=RecordingTable, Image=lambda value: value)
    monkeypatch.setitem(sys.modules, "wandb", fake)
    return fake


def problem():
    return {
        "train": [
            {"input": [[0, 2, 0], [2, 0, 7]], "output": [[2, 0], [0, 2]]},
            {"input": [[7, 0, 2], [0, 7, 0]], "output": [[7, 7]]},
        ],
        "test": [{"input": [[0, 2, 7], [7, 0, 2]]}],
    }


@pytest.mark.parametrize("with_reference", [False, True])
def test_initial_answers_include_training_and_available_full_test_outputs(fake_wandb, with_reference):
    solution = [[[2] * 4 for _ in range(3)]] if with_reference else None
    task = Task("answers", problem(), solution)
    table = training_logging.reference_outputs_table(task)
    assert [row[:2] for row in table.data] == [("train", 1), ("train", 2)] + (
        [("test", 1)] if with_reference else []
    )
    assert table.data[0][5] == problem()["train"][0]["output"]
    if with_reference:
        row = table.data[-1]
        assert row[2:4] == (3, 4)
        assert row[5] == solution[0]
        assert row[4].shape == (3, 4, 3)
        assert task.solution.shape[-2:] != (3, 4)
    else:
        assert task.reference_solutions is None


def test_initial_logging_contains_reference_table_and_problem_image(fake_wandb, monkeypatch):
    task = Task("initial_logging", problem(), None)
    records = []
    monkeypatch.setattr(training_logging, "plot_problem", lambda *a, **k: "problem-figure")
    import matplotlib.pyplot as plt
    monkeypatch.setattr(plt, "close", lambda *a: None)
    training_logging.log_problem(SimpleNamespace(log=records.append), task)
    assert len(records) == 1
    assert records[0]["puzzle/problem"] == "problem-figure"
    assert len(records[0]["puzzle/reference_outputs"].data) == task.n_train


def test_train_predictions_use_output_mode_masks_and_original_colors(fake_wandb):
    task = Task("train_predictions", problem(), None)
    tracker = SolutionTracker(task)
    logits = torch.full((task.n_examples, len(task.colors), task.n_x, task.n_y, 2), -10.0)
    logits[:, task.colors.index(0), :, :, :] = 10
    x_mask = torch.full((task.n_examples, task.n_x, 2), -10.0)
    y_mask = torch.full((task.n_examples, task.n_y, 2), -10.0)
    for index, case in enumerate(problem()["train"]):
        expected = case["output"]
        x0, y0 = (0 if index == 0 else 1), 1
        x_mask[index, x0:x0 + len(expected), 1] = 10
        y_mask[index, y0:y0 + len(expected[0]), 1] = 10
        for row, values in enumerate(expected):
            for column, color in enumerate(values):
                logits[index, :, x0 + row, y0 + column, 1] = -10
                logits[index, task.colors.index(color), x0 + row, y0 + column, 1] = 10
    # Deliberately make one output cell incorrect in the second train example.
    logits[1, :, 1, 1, 1] = -10
    logits[1, task.colors.index(2), 1, 1, 1] = 10
    table = training_logging.training_predictions_table(task, (logits, x_mask, y_mask), tracker)
    assert len(table.data) == 2
    assert table.data[0][3] is True
    assert table.data[0][4] == table.data[0][5] == [[2, 0], [0, 2]]
    assert table.data[1][3] is False
    assert table.data[1][4] == [[7, 7]]
    assert table.data[1][5] == [[2, 7]]
    # RGB arrays use original ARC colors rather than compact model indices.
    np.testing.assert_array_equal(table.data[1][2][0, 0], training_logging.color_list[2])
    assert table.data[0][2].dtype == np.uint8
