import contextlib
import functools
import itertools
import sys
import time
import typing
from importlib.util import find_spec

import numpy as np
import torch

from . import cuda, hooks, module, performance
from ._base import Base
from ._dev_utils._general import _cuda_info, _general_info
from ._version import __version__


class Timer(Base, contextlib.AbstractContextManager):
    r"""**Measure execution time of function.**

    Can be used as context manager or function decorator, perform checkpoints
    or display absolute time from measurements beginning.

    **Used as context manager**::

        with Timer() as timer:
            ... # your operations
            print(timer) # __str__ calls timer.time() internally
            timer.checkpoint() # register checkpoint
            ... # more operations
            print(timer.checkpoint()) # time since last timer.checkpoint() call

        ... # even more operations
        print(timer) # time taken for the block, will not be updated outside of it

    When execution leaves the block, timer will be blocked. Last checkpoint and time taken
    to execute whole block will be returned by `checkpoint()` and `time()` methods respectively.

    **Used as function decorator**::

        @Timer()
        def foo():
            return 42

        value, time = foo()

    Parameters
    ----------
    function : Callable, optional
            No argument function used to measure time. Default: time.perf_counter

    """

    def __init__(self, function: typing.Callable = time.perf_counter):
        self.function = function

        self.start = self.function()
        self.last = self.start
        self.last_checkpoint = self.start

        self.ended: bool = False

    def time(self):
        """**Time taken since the object creation (measurements beginning).**

        Returns
        -------
        time-like
                Whatever `self.function() - self.function()` returns,
                usually fraction of seconds
        """
        if not self.ended:
            return self.function() - self.start
        return self.last - self.start

    def checkpoint(self):
        """**Time taken since last checkpoint call.**

        If wasn't called before, it is the same as as Timer creation time (first call returns
        the same thing as `time()`)

        Returns
        -------
        time-like
                Whatever `self.function() - self.function()` returns,
                usually fraction of seconds
        """
        if not self.ended:
            self.last_checkpoint = self.last
            self.last = self.function()
        return self.last - self.last_checkpoint

    def __call__(self, function):
        @functools.wraps(function)
        def decorated(*args, **kwargs):
            self.start = self.function()
            values = function(*args, **kwargs)
            self.__exit__()
            return values, self.time()

        return decorated

    def __exit__(self, *_, **__) -> None:
        self.last = self.function()
        self.ended: bool = True
        return False

    def __str__(self) -> str:
        return str(self.time())


class seed(Base):
    r"""**Seed PyTorch and numpy.**

    This code is based on PyTorch's reproducibility guide: https://pytorch.org/docs/stable/notes/randomness.html
    Can be used as standard seeding procedure, context manager (seed will be changed only within block) or function decorator.

    **Standard seed**::

            torchfunc.Seed(0) # no surprises I guess

    **Used as context manager**::

        with Seed(1):
            ... # your operations

        print(torch.initial_seed()) # Should be back to seed pre block

    **Used as function decorator**::

        @Seed(1) # Seed only within function
        def foo():
            return 42

    **Important:** It's impossible to put original `numpy` seed after context manager
    or decorator, hence it will be set to original PyTorch's seed.

    Parameters
    ----------
    value: int
            Seed value used in np.random_seed and torch.manual_seed. Usually int is provided
    cuda: bool, optional
            Whether to set PyTorch's cuda backend into deterministic mode (setting cudnn.benchmark to `False`
            and cudnn.deterministic to `True`). If `False`, consecutive runs may be slightly different.
            If `True`, automatic autotuning for convolutions layers with consistent input shape will be turned off.
            Default: `False`

    """

    def __init__(self, value, cuda: bool = False):
        self.value = value
        self.cuda = cuda

        self._last_seed = torch.initial_seed()
        np.random.seed(self.value)
        torch.manual_seed(self.value)

        if self.cuda:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def __enter__(self):
        return self

    def __exit__(self, *_, **__):
        torch.manual_seed(self._last_seed)
        np.random.seed(self._last_seed)
        return False

    def __call__(self, function):
        @functools.wraps(function)
        def decorated(*args, **kwargs):
            value = function(*args, **kwargs)
            self.__exit__()
            return value

        return decorated


def info(general: bool = True, cuda: bool = True) -> str:
    r"""**Return host related info as string.**

    This function may help you tailor your module's architecture to specific environment
    it will be run on.

    For in-depth info regarding possible performance improvements see `torchfunc.performance` submodule.

    **Information is divided into two sections:**

    - general - related to OS, Python version etc.
    - cuda - specific to CUDA hardware

    **Example**::

            print(torchfunc.info(general=False))

    Parameters
    ----------
    general: bool, optional
            Return general informations. Default: `True`
    cuda: bool, optional
            Return CUDA related information. Default: `True`

    Returns
    -------
    str
            Description of system and/or GPU.

    """
    info_string = ""
    if general:
        info_string += _general_info()
        info_string += "\n"
    if cuda:
        info_string += _cuda_info()
    return info_string


def sizeof(obj) -> int:
    r"""**Get size in bytes of Tensor, torch.nn.Module or standard object.**

    Specific routines are defined for torch.tensor objects and torch.nn.Module
    objects. They will calculate how much memory in bytes those object consume.

    If another object is passed, `sys.getsizeof` will be called on it.

    This function works similarly to C++'s sizeof operator.

    **Example**::

        module = torch.nn.Linear(20, 20)
        bias = 20 * 4 # in bytes
        weights = 20 * 20 * 4 # in bytes
        print(torchfunc.sizeof(model) == bias + weights) # True


    Parameters
    ----------
    obj
            Object whose size will be measured.

    Returns
    -------
    int
            Size in bytes of the object

    """
    if torch.is_tensor(obj):
        return obj.element_size() * obj.numel()

    elif isinstance(obj, torch.nn.Module):
        return sum(
            sizeof(tensor)
            for tensor in itertools.chain(obj.buffers(), obj.parameters())
        )
    else:
        return sys.getsizeof(obj)


def installed(module: str) -> bool:
    """**Return True if module is installed.**

    **Example**::

        # Check whether mixed precision library available
        print(torchfunc.installed("apex"))

    Parameters
    ----------
    module: str
            Name of the module to be checked.

    Returns
    -------
    bool
            True if installed.

    """
    return find_spec(module) is not None
