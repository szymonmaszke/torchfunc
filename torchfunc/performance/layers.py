r"""
**Check any performance caveats related to PyTorch and it's layers.**

Using functionalities below you can check whether your architecture follows
current good practices related to performance of `torch.nn.Module` concrete layers.

"""

import abc
import collections
import sys
import typing

import torch

from .._base import Base


class Depthwise(Base):
    r"""**Check whether any convolution layer is a so-called depthwise convolution.**

    Depthwise convolution is faster for images with input data in format
    (batch, height, width, channel) as specialized kernels are available.

    Currently PyTorch does not support this functionality, so using those may actually
    slow down your neural network.

    Depthwise convolution might still be useful in order to save memory, not so
    performance-wise.

    For easy to follow guidelines, use `tips` method of this class.

    Example::

        model = torch.nn.Sequential(
            torch.nn.Conv1d(64, 64, kernel_size=3, groups=64),
            torch.nn.Conv2d(3, 32, kernel_size=3, groups=1),
            torch.nn.Conv2d(32, 32, kernel_size=3, groups=32),
        )
        for index in torchfunc.performance.layers.Depthwise().children(model):
            print(index) # Should print 0 and 2

    Attributes
    ----------
    checkers : Tuple[Callable], optional
            Functions checking whether given module is depthwise convolution.
            Should return True in such case, False otherwise.
            Default: `Depthwise.default_checker`; if module's groups count is equal
            to module's `in_channels` True is returned. Works for PyTorch's `ConvNd` layers.

    """

    def __init__(
        self, checkers: typing.Tuple[typing.Callable[[torch.nn.Module], bool]] = None
    ):
        self.checkers: typing.Tuple[typing.Callable] = (
            Depthwise.default_checker,
        ) if checkers is None else checkers

    @classmethod
    def default_checker(cls, module):
        r"""**Default checking method suitable for PyTorch's built-in convolution layers.**

        Checks whether count of groups is equal to count of in_channels.

        **Important:**

        If you want to provide custom checker, you should return `True`
        (module being depthwise convolution) or `False` for any
        module that is passed to this function.


        Parameters
        ----------
        module : torch.nn.Module
                Module (or submodule) for which True means it's depthwise.

        Returns
        ----------
        List[int]
                Submodule's indices where depthwise convolution was located.
        """
        if hasattr(module, "groups") and hasattr(module, "in_channels"):
            return module.groups == module.in_channels and module.in_channels != 1
        return False

    def _analyse(self, module, function):
        for index, submodule in enumerate(getattr(module, function)()):
            for checker in self.checkers:
                if checker(submodule):
                    yield index

    def modules(self, module: torch.nn.Module):
        r"""**Look for Depthwise convolution using** `modules()` **method (recursive scanning).**

        Parameters
        ----------
        module : torch.nn.Module
                Module to be scanned

        Yields
        ------
        int
                Indices where module is considered depthwise convolution.
        """

        yield from self._analyse(module, "modules")

    def children(self, module: torch.nn.Module):
        r"""**Look for Depthwise convolution using module's** `children()` **method (shallow scanning).**

        Parameters
        ----------
        module : torch.nn.Module
                Module to be scanned

        Yields
        ------
        int
                Indices where module is considered depthwise convolution.
        """
        yield from self._analyse(module, "children")

    def tips(self, module: torch.nn.Module) -> str:
        r"""**Return** `str` **representation of** `modules()` **method.**

        It is advised to use this function to get tips in order to easily fix
        performance issues related to depthwise convolution.

        Parameters
        ----------
        module : torch.nn.Module
                Module to be scanned

        Returns
        -------
        str
                String representing tips related to depthwise convolution.
        """
        depthwise = self.modules(module)
        if depthwise:
            return (
                "Depthwise convolutions are not currently using specialized kernel and might be slower.\n"
                + "See this issue: https://github.com/pytorch/pytorch/issues/18631 for more information.\n"
                + "Indices of those modules:\n"
                + str(list(depthwise))
                + "\nYou may want to decrease number of groups (like it's done for ResNeXt) for possible speed & accuracy improvements."
            )
        return ""


class Inplace(Base):
    r"""**Check whether any submodule/child of module is set to inplace mode.**

    Inplace operations may interfere with traced module (kernel fusion) and cause slowdowns.
    See `this issue <https://github.com/pytorch/pytorch/issues/23655>`__ for more information.

    **Example**::

        model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=3, groups=64),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(64, 64, kernel_size=3),
            torch.nn.ReLU6(inplace=True),
            torch.nn.Conv2d(64, 128, kernel_size=3, groups=32),
        )

        for index in torchfunc.performance.layers.Inplace().children(model):
            print(index) # Should print 1 and 3

    For easy to follow guidelines, use `tips` method of this class.

    Attributes
    ----------
    attribute: Tuple[str], optional
            Attributes names indicating whether current op is inplace. Do not specify if you are not using
            custom modules not following pytorch's conventions. Default: `("inplace",)`.
            Existence of all those attributes will be checked in module. If any of them exists
            and is `True`, it will be considered as inplace operation.

    """

    def __init__(self, inplace: typing.Tuple[str] = ("inplace",)):
        self.inplace = inplace

    def _analyse(self, module: torch.nn.Module, method: str):
        for index, submodule in enumerate(getattr(module, method)()):
            for attribute in self.inplace:
                if hasattr(submodule, attribute):
                    if getattr(submodule, attribute):
                        yield index

    def modules(self, module: torch.nn.Module):
        r"""**Look for inplace operation using** `modules()` **method (recursive scanning).**

        Yields
        ------
        int
                Indices where module is probably `inplace`.
        """
        yield from self._analyse(module, "modules")

    def children(self, module: torch.nn.Module):
        r"""**Look for inplace operation using** `children()` **method (shallow scanning).**

        Yields
        ------
        int
                Indices where module is probably `inplace`.
        """
        yield from self._analyse(module, "children")

    def tips(self, module: torch.nn.Module) -> str:
        r"""**Return** `str` **representation of** `modules()` **method.**

        It is advised to use this function to get tips in order to easily fix
        performance issues related to inplace operations.

        Parameters
        ----------
        module : torch.nn.Module
                Module to be scanned

        Returns
        -------
        str
                String representing tips related to inplace operations.
        """
        inplace = self.modules(module)
        if inplace:
            return (
                "In-place operations might harm kernel fusion. Indices of those modules:\n"
                + str(list(inplace))
                + "\nYou may want to remove inplace flag (see this issue: https://github.com/pytorch/pytorch/issues/23655)"
            )
        return ""
