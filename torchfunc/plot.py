r"""
**This module allows you to plot and visualize specific parts of your neural network.**

Currently you can display linear activations and their strength using `Activations1d`.
For future plans on visualization see `ROADMAP` document in GitHub repository.

"""

import builtins
import dataclasses
import typing

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

from ._base import Base


@dataclasses.dataclass(repr=False)
class Activations1d(Base):
    """**Plot activations given as 1D tensors.**

    One can easily visualize activations of torch.nn.Linear module
    (or any other returning 1d Tensor as output from the layer)
    Activations can be recorded with record module.

    Can be used in conjunction with `torchfunc.record` module.

    Example::

        import matplotlib.pyplot as plt

        ... # Define recorder previously
        fig = plt.figure(figsize=(12, 12))
        ax = fig.gca()

        plotter = torchfunc.plot.Activations1d(ax=ax)
        # Pass 1d tensors, here from recorder
        plotter(*[data for data in recorder]) # See recorder module
        fig.savefig(f"network_activations.png")

    Attributes
    ----------
    ax : matplotlib.axis.Axis
        Matplotlib axis where activations will be plotted
    colormap : matplotlib.colors.Color, optional
        Instance of colormap. See https://matplotlib.org/3.1.1/gallery/color/colormap_reference.html.
        You can use `matplotlib.pyplot.get_cmap(name)` to get cmap.
        Default: `None` (default colormap)
    horizontal : bool, optional
        Whether network should be drawn horizontally. Default: False
    value_max : int, optional
        Maximum value of neuron activation that will be displayed on colorbar.
        If not specified, it will be taken from data passed via __call__.
    value_min : int, optional
        Minimum value of neuron activation that will be displayed on colorbar.
        If not specified, it will be taken from data passed via __call__.
        Default: None
    neuron_size : int, optional
        Size of single neuron. It is the same as size argument of `matplotlib.pyplot.scatter`'s `s` (size) argument.
        Default: None
    edge_colors : typing.Any, optional
        Edge colors of neurons. It is the same as `edge_colors` argument of `matplotlib.pyplot.scatter`.
        Default: None
    orientation : str, optional
        Location of colorbar. Same as `matplotlib.pyplot.colorbar`'s orientation argument.
        Default: vertical

    """

    ax: matplotlib.axis.Axis
    colormap: typing.Any = None
    horizontal: bool = False
    value_max: int = None
    value_min: int = None
    neuron_size: int = None
    edge_colors: typing.Any = None
    orientation: str = "vertical"

    def __post_init__(self):
        self.ax.axis("off")

    def __call__(self, *activations: torch.Tensor) -> None:
        def _find_extremes(activations, initial_value, function: str):
            return (
                getattr(builtins, function)(map(getattr(torch, function), activations))
                if initial_value is None
                else initial_value
            )

        if activations:
            vmin, vmax = (
                _find_extremes(activations, self.value_min, "min"),
                _find_extremes(activations, self.value_max, "max"),
            )
            longest_layer = max(map(len, activations))

            last_scatter = None
            for index, layer in enumerate(activations):
                spacing = int((longest_layer - len(layer)) * 0.5)
                layer_position_in_model = np.full(len(layer), index)
                neuron_positions = range(spacing, spacing + len(layer))
                last_scatter = self.ax.scatter(
                    x=neuron_positions if self.horizontal else layer_position_in_model,
                    y=layer_position_in_model if self.horizontal else neuron_positions,
                    c=layer.float().numpy(),
                    cmap=self.colormap,
                    s=self.neuron_size,
                    vmin=vmin,
                    vmax=vmax,
                    edgecolors=self.edge_colors,
                )

            plt.colorbar(last_scatter, ax=self.ax, orientation=self.orientation)
