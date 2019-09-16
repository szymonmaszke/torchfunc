r"""
**This module provides functionalities related to torch.nn.Module instances (e.g. freezing parameters).**

For performance analysis of `torch.nn.Module` please see subpackage `performance`.

"""

import copy
import datetime
import pathlib

import torch

from ._base import Base


def _switch(module: torch.nn.Module, weight: bool, bias: bool, value: bool):
    for name, param in module.named_parameters():
        if bias and weight:
            param.requires_grad_(value)
        elif ("bias" in name and bias) or ("weight" in name and weight):
            param.requires_grad_(value)

    return module


def freeze(
    module: torch.nn.Module, weight: bool = True, bias: bool = True
) -> torch.nn.Module:
    r"""**Freeze module's parameters.**

    Sets `requires_grad` to `False` for specified parameters in module.
    If bias and weight are specified, ALL parameters will be frozen (even if their names
    are not matched by `weight` and bias).

    If you want to freeze only those whose names contain `bias` or `weight`,
    call the function twice consecutively (once with `bias=True` and `weight=False` and vice versa).

    Example::

        logistic_regression = torch.nn.Sequential(
            torch.nn.Linear(784, 10),
            torch.nn.Sigmoid(),
        )

        # Freeze only bias in logistic regression
        torchfunc.freeze(logistic_regression, weight = False)


    Parameters
    ----------
    module : torch.nn.Module
            Module whose weights and biases will be frozen.
    weight : bool, optional
            Freeze weights. Default: True
    bias : bool, optional
            Freeze bias. Default: True

    Returns
    -------
    module : torch.nn.Module
        Module after parameters were frozen

    """
    return _switch(module, weight, bias, value=False)


def unfreeze(
    module: torch.nn.Module, weight: bool = True, bias: bool = True
) -> torch.nn.Module:
    r"""**Unfreeze module's parameters.**

    Sets `requires_grad` to `True` for all parameters in module.
    Works as complementary function to freeze, see it's documentation.

    Parameters
    ----------
    module : torch.nn.Module
            Module whose weights and biases will be unfrozen.
    weight : bool, optional
            Freeze weights. Default: True
    bias : bool, optional
            Freeze bias. Default: True

    Returns
    -------
    module : torch.nn.Module
        Module after parameters were unfrozen

    """
    return _switch(module, weight, bias, value=False)


class Snapshot(Base):
    r"""**Save module snapshots in memory and/or disk.**

    Next models can be added with `+` or `+=` and their state or whole model saved
    to disk with appropriate methods.

    All added modules are saved unless removed with `pop()` method.

    Example::

        snapshot = torchfunc.module.Snapshot()
        snapshot += torch.nn.Sequential(torch.nn.Linear(784, 10), torch.nn.Sigmoid())
        snapshot.save() # Saved last model to disk with default settings

    Parameters
    ----------
    *modules : torch.nn.Module
        Var args of PyTorch modules to be kept.

    """

    def __init__(self, *modules: torch.nn.Module):
        if modules:
            self.modules = list(modules)
        self.modules = []
        self.timestamps = []

    def __iadd__(self, other):
        self.modules.append(other)
        self.timestamps.append(datetime.datetime.now())
        return self

    def __radd__(self, other):
        return self + other

    def __add__(self, other: torch.nn.Module):
        new = Snapshot(copy.deepcopy(self.modules))
        new += other
        return new

    def pop(self, index: int = -1):
        r"""**Remove module at `index` from memory.**


        Parameters
        ----------
        index : int, optional
            Index of module to be removed. Default: -1 (last module)

        Returns
        ----------
        module : torch.nn.Module
            Module removed by this operation

        """
        return self.modules.pop(index)

    def _save(
        self, index, folder: pathlib.Path, name: pathlib.Path, remove: bool, state: bool
    ) -> None:
        module = self.modules[index]
        if state:
            module = module.state_dict()
        if folder is None:
            folder = pathlib.Path(".")
        if name is None:
            name = pathlib.Path(
                f"module_" + "state_"
                if state
                else "" + f"{len(self.modules)}_{self.timestamps[index]}.pt"
            )
        if remove:
            self.pop(index)
        torch.save(module, folder / name)

    def save(
        self,
        folder: pathlib.Path = None,
        name: pathlib.Path = None,
        index: int = -1,
        remove: bool = False,
    ) -> None:
        r"""**Save module to disk.**

        See PyTorch's docs for more information: https://pytorch.org/tutorials/beginner/saving_loading_models.html#save-load-entire-model

        Parameters
        ----------
        folder : pathlib.Path, optional
            Name of the folder where model will be saved. Defaults to current working directory.
        name : pathlib.Path, optional
            Name of the file. Default: module_{index}_{timestamp}.pt
        index : int, optional
            Index of the module to be saved. Default: Last module
        remove : bool, optional
            Whether module should be removed from memory after saving. Useful
            for keeping only best/last model in memory. Default: False

        """
        self._save(index, folder, name, remove, state=False)

    def save_state(
        self,
        folder: pathlib.Path = None,
        name: pathlib.Path = None,
        index: int = -1,
        remove: bool = False,
    ) -> None:
        r"""**Save module's state to disk.**

        See PyTorch's docs on `state_dict` for more information: https://pytorch.org/tutorials/beginner/saving_loading_models.html#save-load-state-dict-recommended

        Parameters
        ----------
        folder : pathlib.Path, optional
            Name of the folder where model will be saved. Defaults to current working directory.
        name : pathlib.Path, optional
            Name of the file. Default: module_{index}_{timestamp}.pt
        index : int, optional
            Index of the module to be saved. Default: Last module
        remove : bool, optional
            Whether module should be removed from memory after saving. Useful
            for keeping only best/last model in memory. Default: False

        """
        self._save(index, folder, name, remove, state=True)
