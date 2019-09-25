r"""
**This module allows you for easier hook registration (e.g. based on** `type` **or** `index` **within network).**

Example::

    # Example forward pre hook
    def example_forward_pre(module, inputs):
        return inputs + 1

    # MNIST classifier
    model = torch.nn.Sequential(
        torch.nn.Linear(784, 100),
        torch.nn.ReLU(),
        torch.nn.Linear(100, 50),
        torch.nn.ReLU(),
        torch.nn.Linear(50, 10),
    )
    registrator = torchfunc.hooks.registrators.ForwardPre()
    # Register forwardPreHook for all torch.nn.Linear submodules
    registrator.modules(model, example_forward_pre, types=(torch.nn.Linear))

You could specify indices instead of types (for example all inputs to `torch.nn.Linear` will be registered),
and iterate over `children` instead of `modules`.
"""

import dataclasses
import typing

import torch

from .._base import Base
from ._dev_utils import (children_documentation, modules_documentation,
                         params_documentation, register_condition)


class _Registrator(Base):
    r"""**{}**

    Attributes
    ----------
    handles : List[torch.utils.hooks.RemovableHandle]
        Handles for registered hooks, each corresponds to specific submodule.
        Can be used to unregister certain hooks (though discouraged).

    """

    def __init__(self, register_method, hook):
        self._register_method: typing.Callable = register_method
        self.hook: typing.Callable = hook
        self.handles = []

    def _register_hook(
        self,
        network,
        iterating_function: str,
        types: typing.Tuple[typing.Any] = None,
        indices: typing.List[int] = None,
    ):
        for index, module in enumerate(getattr(network, iterating_function)()):
            if register_condition(module, types, index, indices):
                self.handles.append(getattr(module, self._register_method)(self.hook))

    def __iter__(self):
        return iter(self.handles)

    def __len__(self):
        return len(self.handles)

    def remove(self, index) -> None:
        r"""**Remove hook specified by** `index`.

        Parameters
        ----------
        index: int
            Index of subhook (usually registered for layer)

        """
        handle = self.handles.pop(index)
        handle.remove()

    def modules(
        self,
        module: torch.nn.Module,
        types: typing.Tuple[typing.Any] = None,
        indices: typing.List[int] = None,
    ):
        rf"""**Register** `hook` **using types and/or indices via** `modules` **hook**.

        {modules_documentation()}

        {params_documentation()}

        """
        self._register_hook(module, "modules", types, indices)
        return self

    def children(
        self,
        network,
        types: typing.Tuple[typing.Any] = None,
        indices: typing.List[int] = None,
    ):
        rf"""**Register** `subrecorders` **using types and/or indices via** `children` **hook**.

        {children_documentation()}

        {params_documentation()}

        """
        self._register_hook(network, "children", types, indices)
        return self


@dataclasses.dataclass(repr=False)
class ForwardPre(_Registrator):
    __doc__ = _Registrator.__doc__.format(
        "Register forward pre hook based on module's type or indices."
    )

    hook: typing.Callable

    def __post_init__(self):
        super().__init__("register_forward_pre_hook", self.hook)


@dataclasses.dataclass(repr=False)
class Forward(_Registrator):
    __doc__ = _Registrator.__doc__.format(
        "Register forward hook based on module's type or indices."
    )

    hook: typing.Callable

    def __post_init__(self):
        super().__init__("register_forward_hook", self.hook)


@dataclasses.dataclass(repr=False)
class Backward(_Registrator):
    __doc__ = _Registrator.__doc__.format(
        "Register backward hook based on module's type or indices."
    )

    hook: typing.Callable

    def __post_init__(self):
        super().__init__("register_backward_hook", self.hook)
