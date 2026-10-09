"""Projected D4 block migrated from CompressARC's projected_d4 pilot."""
import torch.nn as nn

from ..helper import apply_residual
from .morphological import D4DirectionShare


class ProjectedD4DirectionShareLayer(nn.Module):
    """x + W2 D4(W1 normalize(x)); one D4 mixer per model depth."""

    def __init__(self, multify, n_layers=1):
        super().__init__()
        self.models = nn.ModuleList([D4DirectionShare() for _ in range(n_layers)])

        def apply_one(dims, x, weights, *, pre_norm=True, use_bias=False, layer_index=0):
            if not dims[2]:
                return x
            axis = sum(dims[:2])
            model = self.models[layer_index]

            def mix(projected):
                if axis == 0:
                    return model(projected.unsqueeze(0)).squeeze(0)
                return model(projected.movedim(axis, 1)).movedim(1, axis)

            return apply_residual(x, weights, mix, pre_norm=pre_norm, use_bias=False)

        self._apply_multitensor = multify(apply_one)

    def forward(self, x, weights, *, pre_norm=True, use_bias=False, layer_index=0):
        if use_bias:
            raise ValueError("Projected D4 direction sharing does not support bias")
        return self._apply_multitensor(
            x, weights, pre_norm=pre_norm, use_bias=False, layer_index=layer_index,
        )
