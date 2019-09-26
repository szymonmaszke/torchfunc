import typing

import torch

from .._base import Base


def register_condition(module, types, index, indices):
    return not (
        ((indices is None) and (types is not None) and (not isinstance(module, types)))
        or ((types is None) and (indices is not None) and (index not in indices))
    )
