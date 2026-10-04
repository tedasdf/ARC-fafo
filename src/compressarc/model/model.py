"""ARC compressor model configuration, parameter construction, and forward pass."""

from config import load_config
from .initializers import Initializer
from .initialization.latent import initialize_latent_weights
from .initialization.layers import initialize_layer_weights
from .initialization.output import initialize_output_weights
from .initialization.symmetry import apply_weight_symmetry
from .layer_factory import LayerFactory
from .multitensor_systems import multify
from ..layers.helper import nonlinear, normalize, softmax
from ..layers.latent_decoder import LatentDecoder
from ..layers.output_heads import OutputHeads
from ..layers.share.primitives import MultitensorShare


class ARCCompressor:
    """The ARC compressor model."""

    def __init__(self, task, config):
        self.multitensor_system = task.multitensor_system
        self.config = config if config is not None else load_config().model
        self.n_layers = self.config.n_layers

        self.initializer = Initializer(
            self.multitensor_system,
            self.channel_dim_fn,
        )

        self._init_latent_weights()
        self._init_layer_weights()
        self._init_output_weights()
        self._apply_symmetry()

        self.weights_list = self.initializer.weights_list
        self._init_layers()

    def channel_dim_fn(self, dims):
        """Return the residual channel width for a multitensor component."""
        return self.config.channel_dim_shared if dims[2] == 0 else self.config.channel_dim_grid

    def _init_latent_weights(self):
        (
            self.multiposteriors,
            self.decode_weights,
            self.target_capacities,
        ) = initialize_latent_weights(self.initializer, self.config)

    def _init_layer_weights(self):
        layer_weights = initialize_layer_weights(self.initializer, self.config)
        for name, weights in layer_weights.items():
            setattr(self, name, weights)

        self.layer_weights = [
            {
                "share_up": self.share_up_weights[index],
                "share_down": self.share_down_weights[index],
                "softmax": self.softmax_weights[index],
                "cummax": self.cummax_weights[index],
                "shift": self.shift_weights[index],
                "direction_share": self.direction_share_weights[index],
                "nonlinear": self.nonlinear_weights[index],
            }
            for index in range(self.n_layers)
        ]

    def _init_output_weights(self):
        self.head_weights, self.mask_weights = initialize_output_weights(
            self.initializer,
            self.channel_dim_fn,
        )

    def _apply_symmetry(self):
        layer_weights = {
            name: getattr(self, name)
            for name in (
                "share_up_weights",
                "share_down_weights",
                "softmax_weights",
                "cummax_weights",
                "shift_weights",
                "direction_share_weights",
                "nonlinear_weights",
            )
        }
        apply_weight_symmetry(
            self.initializer,
            layer_weights,
            self.decode_weights,
        )

    def _init_layers(self):
        factory = LayerFactory()
        self.latent_decoder = LatentDecoder(multify)
        self.output_heads = OutputHeads()
        self.share = MultitensorShare()
        self.cummax = factory.create_cummax(
            self.config.cummax_implementation,
            multify=multify,
        )
        self.shift = factory.create_shift(
            self.config.shift_implementation,
            multify=multify,
        )
        self.direction_share = factory.create_direction_share(
            self.config.direction_share_implementation,
            multify=multify,
        )

    def forward(self):
        x, kl_amounts, kl_names = self.latent_decoder(
            self.target_capacities,
            self.decode_weights,
            self.multiposteriors,
        )

        masks = self.multitensor_system.task.masks
        for layer_num in range(self.n_layers):
            weights = self.layer_weights[layer_num]

            x = self.share.share_up(x, weights["share_up"])
            x = softmax(
                x,
                weights["softmax"],
                pre_norm=True,
                post_norm=False,
                use_bias=False,
            )
            x = self.cummax(
                x,
                weights["cummax"],
                masks,
                pre_norm=False,
                post_norm=True,
                use_bias=False,
            )
            x = self.shift(
                x,
                weights["shift"],
                masks,
                pre_norm=False,
                post_norm=True,
                use_bias=False,
            )
            x = self.direction_share(
                x,
                weights["direction_share"],
                pre_norm=True,
                use_bias=False,
            )
            x = nonlinear(
                x,
                weights["nonlinear"],
                pre_norm=True,
                post_norm=False,
                use_bias=False,
            )
            x = self.share.share_down(x, weights["share_down"])
            x = normalize(x)

        output, x_mask, y_mask = self.output_heads(
            x,
            self.head_weights,
            self.mask_weights,
            self.multitensor_system.task,
        )
        return output, x_mask, y_mask, kl_amounts, kl_names



