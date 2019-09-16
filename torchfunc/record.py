r"""
**This module allows you to record the state of your neural network in various situations
(for example activations value passing through it).**

Recorders are organized similarly to `torch.nn.Module`'s hooks (e.g. `backward`, `forward` and `forward pre`).
Those will record data going out/in from/to `torch.nn.Module` in backward or forward mode.

For example::

    # MNIST classifier
    model = torch.nn.Sequential(
        torch.nn.Linear(784, 100),
        torch.nn.ReLU(),
        torch.nn.Linear(100, 50),
        torch.nn.ReLU(),
        torch.nn.Linear(50, 10),
    )
    # Recorder which sums layer inputs from consecutive forward calls
    recorder = torchfunc.record.ForwardPreRecorder(reduction=lambda x, y: x+y)
    # Record inputs going into Linear(100, 50) and Linear(50, 10)
    recorder.children(model, indices=(2, 3))
    # Train your network normally (or pass data through it)
    ...
    # Save tensors (of shape 100 and 50) in folder, each named 1.pt and 2.pt respectively
    recorder.save(pathlib.Path("./analysis"))

You could specify types instead of indices (all inputs to torch.nn.Linear will be registered),
iterate over modules recursively instead of children etc.

See exact specification below.


"""

import dataclasses
import inspect
import pathlib
import typing

import torch

from ._base import Base


class _Recorder(Base):
    r"""**{}**

    Parameters
    ----------
    condition : Callable, optional
        No argument callable. If True returned, record data.
        Can be used to save data based on external environment (e.g. dataset's label).
        If None, will record every data point. Default: `None`
    reduction : Callable, optional
        Operation to use on incoming data. Should take two arguments, and return one.
        Acts similarly to reduction argument of Python's itertools.reduce.
        If None, data will be added to list, which may be very memory intensive.
        Default: `None`

    Attributes
    ----------
    data : List
        Keeps data passing through hooks, optionally reduced by `reduction`.
        Each item represents data for specified layer
        (with the lower indices being closer to torch.nn.Module's input).
    handles : List[torch.utils.hooks.RemovableHandle]
        Handles for registered hooks, each corresponds to specific submodule.
        Can be used to unregister certain hooks (though discouraged).
    samples : int
        How many samples passed the condition. Useful for calculating running operations
        (e.g. running mean).

    """

    def __init__(self, register_method, method):
        self._register_method: typing.Callable = register_method
        self._method: typing.Callable = method
        self.data = []
        self.hooks = []
        self.handles = []

    def _register_hook(
        self,
        network,
        iterating_function: str,
        types: typing.Tuple[typing.Any] = None,
        indices: typing.List[int] = None,
    ):
        last_index = 0
        for index, module in enumerate(getattr(network, iterating_function)()):
            if isinstance(module, types) or (indices is not None and index in indices):
                hook = self._method(last_index, self.data)
                self.handles.append(getattr(module, self._register_method)(hook))
                self.hooks.append(hook)
                last_index += 1

    def handle(self, index):
        return self.handles[index]

    def samples(self, index):
        return self.hooks[index].samples

    def iter_samples(self):
        for hook in self.hooks:
            yield hook.samples

    def modules(
        self,
        module,
        types: typing.Tuple[typing.Any] = None,
        indices: typing.List[int] = None,
    ):
        r"""**Recursively register gathering hook for specified module, types and/or indices.**

        This reduction will use `modules` method of `torch.nn.Module` to iterate over available submodules.
        If you wish to iterate non-recursively, use `children`.

        Parameters
        ----------
        module : torch.nn.Module
            Module (usually neural network) for which inputs will be collected.
        types : Tuple[typing.Any], optional
            Types whose input's will be recorded. E.g. [torch.nn.Conv2d, torch.nn.Linear] will register
            hooks on every module module of those types. By default, no hook will be registered based on type.
            If you want to register input to every module/child, use object base class.
            Default: `None`
        indices : Iterable[int], optional
            Indices of modules whose inputs will be registered.
            Default: `None` (no modules will have hooks registered based on index).

        Returns
        -------
        self

        """
        self._register_hook(module, "modules", types, indices)
        return self

    def children(
        self,
        network,
        types: typing.Tuple[typing.Any] = None,
        indices: typing.List[int] = None,
    ):

        r"""**Non-recursively register gathering hook for specified module, types and/or indices.**

        This reduction will use `children` method of `torch.nn.Module` to iterate over available submodules.
        This is a shallow iteration (only submodules of current module).

        Parameters
        ----------
        network: torch.nn.Module
                Module (usually neural network) for which inputs will be collected.
        types: typing.Tuple[typing.Any], optional
                Types whose input's will be recorded. E.g. [torch.nn.Conv2d, torch.nn.Linear] will register
                hooks on every module module of this type. By default, no hook will be registered based on type.
                If you want to register input to every module/child, use object base class.
                Default: None
        indices: typing.Iterable[int], optional
                Indices of modules whose inputs will be registered.
                Default: None (no modules will have hooks registered based on index).

        Returns
        -------
        self

        """
        self._register_hook(network, "children", types, indices)
        return self

    def save(self, path: pathlib.Path, mkdir: bool = False, *args, **kwargs):
        """Save data tensors to specified path.

        Each data tensor will be indexed by integer `[0, N)`, where those with smaller
        indices are closer to the input. Ordering is the same as in `__getitem__` or `__iter__`.

        Parameters
        ----------
        path: pathlib.Path
                Path where tensors will be saved.
        mkdir: bool, optional
                If True, create directory if doesn't exists. Default: False
        *args:
                Varargs passed to pathlib.Path's mkdir method if mkdir argument set to True.
        *kwargs:
                Kwarargs passed to pathlib.Path's mkdir method if mkdir argument set to True.

        """
        if mkdir:
            path.mkdir(*args, **kwargs)
        for index, module in enumerate(self):
            torch.save(module, path / f"{index}.pt")

    def __setitem__(self, index, item):
        self.data[index] = item

    def __getitem__(self, index):
        return self.data[index]

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)


@dataclasses.dataclass
class _Hook:
    index: int
    data: typing.List
    samples: int = 0

    def _call(self, to_record, condition, reduction):
        if condition() or condition is None:
            self.samples += 1
            if self.index >= len(self.data):
                self.data.append(to_record[0])
                if reduction is None:
                    self.data[-1] = [self.data[-1]]
            else:
                if reduction is not None:
                    self.data[self.index] = reduction(
                        self.data[self.index], to_record[0]
                    )
                else:
                    self.data[self.index].append(to_record[0])


@dataclasses.dataclass(repr=False)
class ForwardPreRecorder(_Recorder):
    __doc__ = _Recorder.__doc__.format(
        "Record input values before forward of specified layer(s)."
    )

    condition: typing.Callable = None
    reduction: typing.Callable = None

    def __post_init__(self):
        class ForwardPreHook(_Hook):
            def __call__(inner_self, module, inputs):
                inner_self._call(inputs, self.condition, self.reduction)

        super().__init__("register_forward_pre_hook", ForwardPreHook)


class ForwardInputRecorder(_Recorder):
    __doc__ = _Recorder.__doc__.format(
        "Record input values after forward of specified layer(s)."
    )

    condition: typing.Callable = None
    reduction: typing.Callable = None

    def __post_init__(self):
        class ForwardInputHook(_Hook):
            def __call__(inner_self, module, inputs, _):
                inner_self._call(inputs, self.condition, self.reduction)

        super().__init__("register_forward_hook", ForwardInputHook)


class ForwardOutputRecorder(_Recorder):
    __doc__ = _Recorder.__doc__.format(
        "Record output values after forward of specified layer(s)."
    )

    condition: typing.Callable = None
    reduction: typing.Callable = None

    def __post_init__(self):
        class ForwardOutputHook(_Hook):
            def __call__(inner_self, module, _, outputs):
                inner_self._call(outputs, self.condition, self.reduction)

        super().__init__("register_forward_hook", ForwardOutputHook)


class BackwardInputRecorder(_Recorder):
    __doc__ = _Recorder.__doc__.format(
        "Record input gradients after those are calculated w.r.t. specified module."
    )

    condition: typing.Callable = None
    reduction: typing.Callable = None

    def __post_init__(self):
        class BackwardInputHook(_Hook):
            def __call__(inner_self, module, grad_inputs, _):
                inner_self._call(grad_inputs, self.condition, self.reduction)

        super().__init__("register_backward_hook", BackwardInputHook)


class BackwardOutputRecorder(_Recorder):
    __doc__ = _Recorder.__doc__.format(
        "Record output gradients after those are calculated w.r.t. specified module."
    )

    condition: typing.Callable = None
    reduction: typing.Callable = None

    def __post_init__(self):
        class BackwardOutputHook(_Hook):
            def __call__(inner_self, module, _, outputs):
                inner_self._call(outputs, self.condition, self.reduction)

        super().__init__("register_backward_hook", BackwardOutputHook)
