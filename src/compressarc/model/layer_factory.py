"""Factories for selecting concrete layer implementations."""

from ..layers.shift.primitives import ShiftPrimitives
from ..layers.cummax.primitives import CummaxPrimitives
from ..layers.direction_share.primitives import DirectionSharePrimitives


class LayerFactory:
    """Construct configured layer implementations from a small registry.

    New implementations can be registered under a name and selected from model
    configuration without changing the model's forward-pass structure.
    """

    def __init__(self):
        self._implementations = {
            "shift": {"primitives": ShiftPrimitives},
            "cummax": {"primitives": CummaxPrimitives},
            "direction_share": {"primitives": DirectionSharePrimitives},
        }

    def register(self, layer_name, implementation_name, implementation):
        """Register a layer class for a layer/implementation name pair."""
        if not callable(implementation):
            raise TypeError("implementation must be a class or callable constructor")
        self._implementations.setdefault(layer_name, {})[implementation_name] = implementation

    def create(self, layer_name, implementation_name, **kwargs):
        """Construct the requested implementation, reporting valid choices on error."""
        implementations = self._implementations.get(layer_name, {})
        try:
            implementation = implementations[implementation_name]
        except KeyError as exc:
            choices = ", ".join(sorted(implementations)) or "none registered"
            raise ValueError(
                f"Unknown {layer_name!r} implementation {implementation_name!r}; "
                f"available choices: {choices}"
            ) from exc
        return implementation(**kwargs)

    def create_direction_share(self, implementation_name="primitives", **kwargs):
        """Construct the selected direction-sharing implementation."""
        return self.create("direction_share", implementation_name, **kwargs)

    def create_cummax(self, implementation_name="primitives", **kwargs):
        """Construct the selected cummax implementation."""
        return self.create("cummax", implementation_name, **kwargs)

    def create_shift(self, implementation_name="primitives", **kwargs):
        """Construct the selected shift implementation."""
        return self.create("shift", implementation_name, **kwargs)