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

    Next modules can be added with `+` or `+=` and their state or whole model saved
    to disk with appropriate methods.

    All added modules are saved unless removed with `pop()` method.

    Additionally, self-explainable methods like `len`, `__iter__` or item access
    are provided (although there is no `__setitem__` as it's discouraged
    to mutate contained modules).

    Example::

        snapshot = torchfunc.module.Snapshot()
        snapshot += torch.nn.Sequential(torch.nn.Linear(784, 10), torch.nn.Sigmoid())
        snapshot.save("models") # Save all modules to models folder

    Parameters
    ----------
    *modules : torch.nn.Module
        Var args of PyTorch modules to be kept.

    """

    def __init__(self, *modules: torch.nn.Module):
        if modules:
            self.modules = list(modules)
        else:
            self.modules = []
        self.timestamps = []

    def __len__(self):
        return len(self.modules)

    def __iter__(self):
        return iter(self.modules)

    def __getitem__(self, index):
        return self.modules[index]

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
        r"""**Remove module at** `index` **from memory.**

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

    def _save(self, folder: pathlib.Path, remove: bool, state: bool, *indices) -> None:
        if folder is None:
            folder = pathlib.Path(".")

        def _single_save(index):
            module = self.modules[index]
            if state:
                module = module.state_dict()
            name = pathlib.Path(
                f"module_" + "state_"
                if state
                else "" + f"{len(self.modules)}_{self.timestamps[index]}.pt"
            )
            if remove:
                self.pop(index)
            torch.save(module, folder / name)

        if not indices:
            indices = range(len(self))

        for index in indices:
            _single_save(index)

    def save(
        self, folder: pathlib.Path = None, remove: bool = False, *indices: int
    ) -> None:
        r"""**Save module to disk.**

        Snapshot(s) will be saved using the following naming convention::

            module_{index}_{timestamp}.pt

        See `PyTorch's docs <https://pytorch.org/tutorials/beginner/saving_loading_models.html#save-load-entire-model>`__
        for more information.

        Parameters
        ----------
        folder : pathlib.Path, optional
            Name of the folder where model will be saved. It has to exist.
            Defaults to current working directory.
        remove : bool, optional
            Whether module should be removed from memory after saving. Useful
            for keeping only best/last model in memory. Default: `False`
        *indices: int, optional
            Possibly empty varargs containing indices of modules to be saved.
            Negative indexing is supported.
            If empty, save all models.

        """
        self._save(folder, remove, state=False, *indices)

    def save_state(
        self, folder: pathlib.Path = None, remove: bool = False, *indices: int
    ) -> None:
        r"""**Save module's state to disk.**

        Snapshot(s) will be saved with using the following naming convention::

            module_{index}_{timestamp}.pt

        See PyTorch's docs about `state_dict <https://pytorch.org/tutorials/beginner/saving_loading_models.html#save-load-state-dict-recommended>`__ for more
        information.

        Parameters
        ----------
        folder : pathlib.Path, optional
            Name of the folder where model will be saved. It has to exist.
            Defaults to current working directory.
        remove : bool, optional
            Whether module should be removed from memory after saving. Useful
            for keeping only best/last model in memory. Default: False
        *indices: int, optional
            Possibly empty varargs containing indices of modules to be saved.
            Negative indexing is supported.
            If empty, save all models.

        """
        self._save(folder, remove, state=True, *indices)
