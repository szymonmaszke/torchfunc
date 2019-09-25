import typing

import torch

from .._base import Base


def register_condition(module, types, index, indices):
    return not (
        ((indices is None) and (types is not None) and (not isinstance(module, types)))
        or ((types is None) and (indices is not None) and (index not in indices))
    )


def children_documentation():
    return "This function will use `children` method of `torch.nn.Module` to iterate over available submodules. If you wish to iterate recursively, use `modules`."


def modules_documentation():
    return "This function will use `modules` method of `torch.nn.Module` to iterate over available submodules. If you wish to iterate non-recursively, use `children`."


def params_documentation():
    return r"""

    **Important:**

    If `types` and `indices` are left with their default values, all modules
    will have `subrecorders` registered.

    Parameters
    ----------
    module : torch.nn.Module
        Module (usually neural network) for which inputs will be collected.
    types : Tuple[typing.Any], optional
        Module types for which data will be recorded. E.g. `(torch.nn.Conv2d, torch.nn.Linear)`
        will register `subrecorders` on every module being instance of either `Conv2d` or `Linear`.
        Default: `None`
    indices : Iterable[int], optional
        Indices of modules whose inputs will be registered.
        Default: `None`

    Returns
    -------
    self
    """
