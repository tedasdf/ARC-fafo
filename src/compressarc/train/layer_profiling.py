"""Layer labels only while the short operator capture is active."""
from contextlib import nullcontext
from contextvars import ContextVar
import torch

active_layer_profile = ContextVar('active_layer_profile', default=False)


def profile_layer(name):
    if active_layer_profile.get():
        return torch.profiler.record_function('layer/' + name)
    return nullcontext()
