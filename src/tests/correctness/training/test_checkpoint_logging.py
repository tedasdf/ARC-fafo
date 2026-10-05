"""Check model artifacts and checkpoint upload before run closure."""

from argparse import Namespace

import pytest
import torch

from config import load_config
from preprocessing import Task
from compressarc.train.logging import log_model_checkpoint


class RecordingRun:
    id = 'checkpoint-test'

    def __init__(self):
        self.events = []
        self.artifact = None

    def log_artifact(self, artifact, aliases):
        assert 'finish' not in self.events
        self.events.append('upload')
        self.artifact = artifact
        self.aliases = aliases
        return artifact

    def finish(self):
        self.events.append('finish')


def test_log_model_checkpoint_artifact(tmp_path):
    path = tmp_path / 'task.pt'
    torch.save({'weights': [torch.tensor([1.0])]}, path)
    run = RecordingRun()
    result = log_model_checkpoint(run, path, metadata={'task_name': 'task'})
    assert result.type == 'model'
    assert result.name == 'model-checkpoint-test'
    assert result.metadata == {'task_name': 'task'}
    assert set(result.manifest.entries) == {'task.pt'}
    assert run.aliases == ['latest']


def test_checkpoint_logging_disabled_needs_no_file(tmp_path):
    assert log_model_checkpoint(None, tmp_path / 'missing.pt') is None


def test_trainer_saves_and_uploads_before_finishing(tmp_path, monkeypatch):
    import train

    config = load_config(overrides=[
        'training.iterations=1', 'training.device=cpu', 'model.n_layers=1',
        'model.channel_dim_shared=2', 'model.channel_dim_grid=2',
        'model.share_up_dim=2', 'model.share_down_dim=2', 'model.decoding_dim=2',
        'model.softmax_dim=2', 'model.cummax_dim=2', 'model.shift_dim=2',
        'model.nonlinear_dim=2', 'model.direction_share_implementation=d4',
    ])
    problem = {'train': [{'input': [[0, 1], [1, 0]], 'output': [[0, 1], [1, 0]]}],
               'test': [{'input': [[1, 0], [0, 1]]}]}
    task = Task('upload_test', problem, None)
    run = RecordingRun()
    monkeypatch.setattr(train, 'parse_args', lambda: Namespace(
        config=None, overrides=[], backend='local', task='upload_test',
        save_checkpoints=True, output_dir=tmp_path,
    ))
    monkeypatch.setattr(train, 'load_config', lambda *a, **k: config)
    monkeypatch.setattr(train.preprocessing, 'preprocess_tasks', lambda *a, **k: [task])
    monkeypatch.setattr(train, 'initialize_wandb', lambda *a: run)
    for name in ['log_problem', 'log_training_step', 'log_final_results', 'log_latent_pca']:
        monkeypatch.setattr(train, name, lambda *a, **k: None)
    original_device = torch.get_default_device()
    try:
        train.main()
    finally:
        torch.set_default_device(original_device)
    checkpoint = torch.load(tmp_path / 'upload_test.pt', weights_only=True)
    assert checkpoint['task'] == 'upload_test'
    assert checkpoint['weights']
    assert all(weight.device.type == 'cpu' for weight in checkpoint['weights'])
    assert checkpoint['config']['model']['direction_share_implementation'] == 'd4'
    assert run.events == ['upload', 'finish']
    assert run.artifact.metadata['iterations_completed'] == 1
    assert set(run.artifact.manifest.entries) == {'upload_test.pt'}
