r"""
**This module allows one to record neural network state (for example when data passes through it).**

`recorders` are organized similarly to
`torch.nn.Module`'s hooks (e.g. `backward`, `forward` and `forward pre`).
Additionally, each can record input or output from specified modules, which
gives us, for example, `ForwardInput` (record input to specified module(s) during forward pass).

Example should make it more clear::

    # MNIST classifier
    model = torch.nn.Sequential(
        torch.nn.Linear(784, 100),
        torch.nn.ReLU(),
        torch.nn.Linear(100, 50),
        torch.nn.ReLU(),
        torch.nn.Linear(50, 10),
    )

    # Recorder which sums layer inputs from consecutive forward calls
    recorder = torchfunc.hooks.recorders.ForwardPre(reduction=lambda x, y: x+y)
    # Record inputs going into Linear(100, 50) and Linear(50, 10)
    recorder.children(model, indices=(2, 3))
    # Train your network normally (pass data through it somehow)
    ...
    # Save tensors (of shape 100 and 50) in folder, each named 1.pt and 2.pt respectively
    recorder.save(pathlib.Path("./analysis"))

You could specify `types` instead of `indices` (for example all forward inputs to `torch.nn.Linear` will be registered),
iterate over modules recursively instead of shallow iteration with `children` method etc.

Each `recorder` has one or more `subrecorders`; those usually correspond to specific layer
for which recording will be done. In the above case, there are two `subrecorders`,
both of `torch.nn.Linear` type.

Additionally one can post-process data contained within `recorder` using `apply`
functionality.

Concrete methods recording different data passing through network are specified below:

"""

import pathlib
import typing

import torch

from .._base import Base
from ._dev_utils import register_condition


class _Recorder(Base):
    r"""**{}**

    You can record only some of the data based on external conditions if `condition`
    `callable` is specified.

    Data can be cumulated together via `reduction` parameter, which is advised
    from the memory perspective.

    You can record an entire batch of data or just a sample via the `data_method` parameter.

    Parameters
    ----------
    condition : Callable, optional
        No argument callable. If True returned, record data.
        Can be used to save data based on external environment (e.g. dataset's label).
        If None, will record every data point. Default: `None`
    reduction : Callable, optional
        Operation to use on incoming data. Should take two arguments, and return one.
        Acts similarly to reduction argument of Python's `functools.reduce <https://docs.python.org/3/library/functools.html#functools.reduce>`__.
        If `None`, data will be added to list, which may be very memory intensive.
        Default: `None`
    data_method : str
        Method used when each hook is called to record the data.
        If 'sample', only the first item in data will be recorded.
        If 'batch', the entire data object will be recorded.
        Default: `'sample'`.
        Choices: `'sample'`, `'batch'`

    Attributes
    ----------
    data : List
        Keeps data passing through subrecorders, optionally reduced by `reduction`.
        Each item represents data for specified `subrecorder`.
    subrecorders: List[Hooks]
        List containing registered subrecorders.
    handles : List[torch.utils.subrecorders.RemovableHandle]
        Handles for registered subrecorders, each corresponds to specific `subrecorder`.
        Can be used to unregister certain subrecorders (though discouraged, please use `remove` method).

    """

    def __init__(self, register_method, method, data_method):
        self._register_method: typing.Callable = register_method
        self._method: typing.Callable = method
        if data_method not in ('sample', 'batch'):
            raise ValueError('Invalid value for parameter `data_method`. Expected either \'sample\' or \'batch\'. You put {}'.format(data_method))
        self.data_method = data_method
        self.data = []
        self.subrecorders = []
        self.handles = []

    def _register_hook(
        self,
        network,
        iterating_function: str,
        types: typing.Tuple[typing.Any] = None,
        indices: typing.Tuple[int] = None,
    ):
        last_index = 0
        for index, module in enumerate(getattr(network, iterating_function)()):
            if register_condition(module, types, index, indices):
                hook = self._method(last_index, self.data, data_method=self.data_method)
                self.handles.append(getattr(module, self._register_method)(hook))
                self.subrecorders.append(hook)
                last_index += 1

    def __setitem__(self, index, item):
        self.data[index] = item

    def __getitem__(self, index):
        return self.data[index]

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.subrecorders)

    def remove(self, index):
        r"""**Remove subrecorder specified by** `index`.

        Subrecorder will not record data passing through it and will be removed
        from `subrecorders` attribute.

        Parameters
        ----------
        index: int
            Index of subrecorder (usually layer)

        Returns
        -------
        torch.Tensor
            Data contained in subrecorder.

        """
        self.handles[index].remove()
        self.subrecorders.pop(index)
        return self.data.pop(index)

    def samples(self, index) -> int:
        r"""**Count of samples passed through subrecorder under** `index`.

        Parameters
        ----------
        index: int
            Index of `subrecorder` (usually layer)

        Returns
        -------
        int
            How many samples passed through specified `subrecorder`.

        """
        return self.subrecorders[index].samples

    def iter_samples(self):
        r"""**Iterate over count of samples for each subrecorder**.

        Parameters
        ----------
        index: int
            Index of subrecorder (usually layer)

        Returns
        -------
        int
            How many samples passed through this subrecorder.

        """
        for hook in self.subrecorders:
            yield hook.samples

    def modules(
        self,
        module: torch.nn.Module,
        types: typing.Tuple[typing.Any] = None,
        indices: typing.List[int] = None,
    ):
        r"""**Register** `subrecorders` **using types and/or indices via** `modules` **method**.

        This function will use `modules` method of `torch.nn.Module` to iterate over available submodules. If you wish to iterate non-recursively, use `children`.

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

        self._register_hook(module, "modules", types, indices)
        return self

    def children(
        self,
        network,
        types: typing.Tuple[typing.Any] = None,
        indices: typing.List[int] = None,
    ):
        r"""**Register** `subrecorders` **using types and/or indices via** `children` **method**.

        This function will use `children` method of `torch.nn.Module` to iterate over available submodules. If you wish to iterate recursively, use `modules`.

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

        self._register_hook(network, "children", types, indices)
        return self

    def save(self, path: pathlib.Path, mkdir: bool = False, *args, **kwargs):
        """**Save data tensors within specified path.**

        Each data tensor will be indexed by integer `[0, N)`, where indices
        represent consecutive `subrecorders`.

        Parameters
        ----------
        path: pathlib.Path
                Path where tensors will be saved.
        mkdir: bool, optional
                If True, create directory if doesn't exists. Default: False
        *args:
                Varargs passed to `pathlib.Path`'s `mkdir` method if `mkdir` argument set to True.
        *kwargs:
                Kwarargs passed to `pathlib.Path`'s `mkdir` method if `mkdir` argument set to True.

        """
        if mkdir:
            path.mkdir(*args, **kwargs)
        for index, subrecorder in enumerate(self):
            torch.save(subrecorder, path / "{}.pt".format(index))

    def apply(self, function: typing.Callable):
        """**Apply function to data contained in each subrecorder.**

        Data will be modified an saved inside data of each subrecorder.
        This function may make `recorder` unusable, it's up to user
        to ensure correct functioning after this functionality was used.

        Parameters
        ----------
        function: Callable
                Single argument (`torch.Tensor` data from `subrecorder`) callable
                returning anything.

        """
        for subrecorder in self:
            subrecorder = function(subrecorder)

    def apply_sample(self, function: typing.Callable) -> None:
        """**Apply function to data contained in each subrecorder.**

        Works like `apply`, except `Callable` is passed number of samples passed
        through `subrecorder` as second argument

        Parameters
        ----------
        function: Callable
                Two argument (`torch.Tensor` data from `subrecorder` and number of `samples` which passed through it)
                `Callable` returning anything.

        """
        for subrecorder, sample in zip(self, self.iter_samples()):
            subrecorder = function(subrecorder, sample)

    def set_data_method(self, data_method: str):
        r"""**Set each subrecorder's** `data_method` **.**

        Parameters
        ----------
        data_method: str
            Data method of subrecorder (either `'sample'` or `'batch'`)

        """
        if data_method not in ('sample', 'batch'):
            raise ValueError('Invalid value for parameter `data_method`. Expected either \'sample\' or \'batch\'. You put {}'.format(data_method))
        
        self.data_method = data_method

        for hook in self.subrecorders:
            hook.data_method = self.data_method


class _Hook:
    def __init__(self, index: int, data: typing.List, samples: int = 0, data_method: str = 'sample'):
        self.index = index
        self.data = data
        self.samples = samples
        self.data_method = data_method

    def _call(self, data, condition, reduction):
        if condition is None or condition():
            if self.data_method == 'sample':
                to_record = data[0]
            elif self.data_method == 'batch':
                to_record = data
            else:
                raise ValueError('Invalid value for parameter `data_method`. Expected either \'sample\' or \'batch\'. You put {}'.format(self.data_method))
            self.samples += 1
            if self.index >= len(self.data):
                self.data.append(to_record)
                if reduction is None:
                    self.data[-1] = [self.data[-1]]
            else:
                if reduction is not None:
                    self.data[self.index] = reduction(
                        self.data[self.index], to_record
                    )
                else:
                    self.data[self.index].append(to_record)


class ForwardPre(_Recorder):
    __doc__ = _Recorder.__doc__.format(
        "Record input values before forward of specified layer(s)."
    )

    def __init__(
        self, condition: typing.Callable = None, reduction: typing.Callable = None, data_method: str = "sample"
    ):
        self.condition = condition
        self.reduction = reduction

        class ForwardPreHook(_Hook):
            def __call__(inner_self, module, inputs):
                inner_self._call(inputs, self.condition, self.reduction)

        super().__init__("register_forward_pre_hook", ForwardPreHook, data_method)


class ForwardInput(_Recorder):
    __doc__ = _Recorder.__doc__.format(
        "Record input values after forward of specified layer(s)."
    )

    def __init__(
        self, condition: typing.Callable = None, reduction: typing.Callable = None, data_method: str = "sample"
    ):
        self.condition = condition
        self.reduction = reduction

        class ForwardInputHook(_Hook):
            def __call__(inner_self, module, inputs, _):
                inner_self._call(inputs, self.condition, self.reduction)

        super().__init__("register_forward_hook", ForwardInputHook, data_method)


class ForwardOutput(_Recorder):
    __doc__ = _Recorder.__doc__.format(
        "Record output values after forward of specified layer(s)."
    )

    def __init__(
        self, condition: typing.Callable = None, reduction: typing.Callable = None, data_method: str = "sample"
    ):
        self.condition = condition
        self.reduction = reduction

        class ForwardOutputHook(_Hook):
            def __call__(inner_self, module, _, outputs):
                inner_self._call(outputs, self.condition, self.reduction)

        super().__init__("register_forward_hook", ForwardOutputHook, data_method)


class BackwardInput(_Recorder):
    __doc__ = _Recorder.__doc__.format(
        "Record input gradients after those are calculated w.r.t. specified module."
    )

    def __init__(
        self, condition: typing.Callable = None, reduction: typing.Callable = None, data_method: str = "sample"
    ):
        self.condition = condition
        self.reduction = reduction

        class BackwardInputHook(_Hook):
            def __call__(inner_self, module, grad_inputs, _):
                inner_self._call(grad_inputs, self.condition, self.reduction)

        super().__init__("register_backward_hook", BackwardInputHook, data_method)


class BackwardOutput(_Recorder):
    __doc__ = _Recorder.__doc__.format(
        "Record output gradients after those are calculated w.r.t. specified module."
    )

    def __init__(
        self, condition: typing.Callable = None, reduction: typing.Callable = None, data_method: str = "sample"
    ):
        self.condition = condition
        self.reduction = reduction

        class BackwardOutputHook(_Hook):
            def __call__(inner_self, module, _, outputs):
                inner_self._call(outputs, self.condition, self.reduction)

        super().__init__("register_backward_hook", BackwardOutputHook, data_method)
