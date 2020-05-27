r"""
**Analyse technological aspects (e.g. compatibility with Tensor Cores) of your module.**

Using functionalities below you can check whether your architecture can use
technology dependent speed improvements.

"""
import collections
import typing

import torch

from .._base import Base

# TO-DO
# https://stackoverflow.com/questions/47913943/is-it-possible-to-see-that-kernel-execution-happened-on-tensor-cores-or-not-via (?)
# Arithmetic Intensity: https://docs.nvidia.com/deeplearning/sdk/dl-performance-guide/index.html#math-mem


class TensorCores(Base):
    r"""**Perform Tensor Cores compatibility tests for given module and it's submodules/children.**

    Interpretation of data returned from this function may pose some problems to users
    unfamiliar with ideas standing behind Tensor Cores.

    Is is advised to use method `tips` to get user friendly information your
    `torch.nn.Module`'s compatitilibty with Tensor Cores.

    Example::

        model = torch.nn.Sequential(
            torch.nn.Linear(128, 100).half(), # Half precision is compatible
            torch.nn.ReLU(),
            torch.nn.Linear(100, 50),
            torch.nn.ReLU(),
            torch.nn.Linear(50, 10).half(),
        )

        analysis = torchscripts.peformance.technology.TensorCores().children(model)
        # Should return dictionary indicating problems with second Linear (wrong shape and type)
        # And last Linear (wrong shape)

    Attributes
    ----------
    linear_types: Tuple[torch.nn.Module], optional
            Tuple of types to be considered linear and which should run with tensor
            cores kernels.

            **Default:** `(torch.nn.Linear, torch.nn.Bilinear)`
    convolution_types: Tuple[torch.nn.Module], optional
            Tuple of types to be considered convolutional and which should run with tensor
            cores kernels.

            **Default:** `(torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d)`
    linear_inputs: Dict[torch.nn.Module, Tuple[str]], optional
            Dict-like where key is the type of module (e.g. `torch.nn.Linear`) and values
            are tuples of attribute names specifying names of input attributes of this type of layer.
            You could use `collections.defaultdict` for easier specification of prevailing attribute names
            like `in_features` for torch.nn.Linear.
            More than one input can be specified, as is the case for `torch.nn.Bilinear`.

            **Default:** `{default_type: ("in_features",), torch.nn.Bilinear: ("in_features1", "in_features2")}`
    linear_outputs: Dict[torch.nn.Module, Tuple[str]], optional
            Dict-like where key is the type of module (e.g. `torch.nn.Linear`) and values
            are tuples of attribute names specifying names of output attributes of this type of layer.
            You could use `collections.defaultdict` for easier specification of prevailing attribute names
            like `out_features` for `torch.nn.Linear`.
            More than one output can be specified, same as `linear_inputs`.

            **Default:** `{default_type: ("out_features",)}`
    convolution_inputs: Dict[torch.nn.Module, Tuple[str]], optional
            Dict-like where key is the type of module (e.g. `torch.nn.Conv2d`) and values
            are tuples of attribute names specifying names of input channels attributes of this type of layer.
            You could use `collections.defaultdict` for easier specification of prevailing attribute names
            like `in_channels` for all torch's convolutions.
            More than one output can be specified, same as `linear_inputs`.

            **Default:** `{default_type: ("in_channels",)}`
    convolution_outputs: Dict[torch.nn.Module, Tuple[str]], optional
            Dict-like where key is the type of module (e.g. torch.nn.Conv2d) and values
            are tuples of attribute names specifying names of output channels attributes of this type of layer.
            You could use collections.defaultdict for easier specification of prevailing attribute names
            like out_channels for all torch's convolutions.
            More than one output can be specified, same as linear_inputs.

            **Default:** `{default_type: ("out_channels",)}`
    float_types: typing.Tuple[types], optional
            Floating point types compatible with TensorCores.

            **Default:** `(torch.half, )`
    integer_types: typing.Tuple[types], optional
            Interger types compatible with TensorCores.

            **Default:** `(torch.short, )`

    """

    def __init__(
        self,
        linear_types=(torch.nn.Linear, torch.nn.Bilinear,),
        convolution_types=(torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d,),
        linear_inputs=None,
        linear_outputs=None,
        convolution_inputs=None,
        convolution_outputs=None,
        float_types=(torch.half,),
        integer_types=(torch.short,),
    ):

        self.linear_types = linear_types
        self.convolution_types = convolution_types
        if linear_inputs is None:
            self.linear_inputs = collections.defaultdict(lambda: ("in_features",))
            self.linear_inputs[torch.nn.Bilinear] = ("in_features1", "in_features2")
        else:
            self.linear_inputs = linear_inputs
        if linear_outputs is None:
            self.linear_outputs = collections.defaultdict(lambda: ("out_features",))
        else:
            self.linear_outputs = linear_outputs
        if convolution_inputs is None:
            self.convolution_inputs = collections.defaultdict(lambda: ("in_channels",))
        else:
            self.convolution_inputs = convolution_inputs
        if convolution_outputs is None:
            self.convolution_outputs = collections.defaultdict(
                lambda: ("out_channels",)
            )
        else:
            self.convolution_outputs = convolution_outputs
        self.float_types = float_types
        self.integer_types = integer_types

    def _analyse(self, module, function: str):
        def _correct_types(data, submodule, index, is_float: bool):
            correct_types = self.float_types if is_float else self.integer_types
            if not any(
                correct_type == submodule.weight.dtype for correct_type in correct_types
            ):
                data["type"]["float" if is_float else "integer"].append(index)

        def _correct_shapes(
            data, submodule, index, attributes, attribute_name, is_float: bool
        ):
            for attribute in attributes[type(submodule)]:
                if hasattr(submodule, attribute):
                    shape = getattr(submodule, attribute)
                    correct = shape % (8 if is_float else 16) == 0
                    if not correct:
                        data["shape"]["float" if is_float else "integer"][
                            attribute_name
                        ].append(index)

        def _find_problems(data, submodule, index, is_float: bool):
            def _operation_problems(operation: str):
                for entry in ("inputs", "outputs"):
                    _correct_shapes(
                        data,
                        submodule,
                        index,
                        getattr(self, operation + "_" + entry),
                        entry,
                        is_float,
                    )

            _correct_types(data, submodule, index, is_float)

            if isinstance(submodule, self.linear_types):
                _operation_problems("linear")
            elif isinstance(submodule, self.convolution_types):
                _operation_problems("convolution")

        #######################################################################
        #
        #                           MAIN FUNCTION
        #
        #######################################################################

        data = {
            "type": {"float": [], "integer": []},
            "shape": {
                "float": {"inputs": [], "outputs": []},
                "integer": {"inputs": [], "outputs": []},
            },
        }

        for index, submodule in enumerate(getattr(module, function)()):
            if hasattr(submodule, "weight"):
                if torch.is_floating_point(submodule.weight):
                    _find_problems(data, submodule, index, is_float=True)
                else:
                    _find_problems(data, submodule, index, is_float=False)

        return data

    def modules(self, module: torch.nn.Module):
        r"""**Check Tensor Cores compatibility using** `modules()` **method (recursive scanning).**

        Parameters
        ----------
        module : torch.nn.Module
                Module to be scanned for Tensor Cores compatibility

        Returns
        -------
        Nested dictionary
                Multilevel dictionary describing modules incompatible with tensor cores.
                First level consists of two fields:

                - `type`: incompatible type with TensorCores
                - `shape`: incompatible types with TensorCores

                Second level for type:

                - `float`: module is floating point type but it's type is incompatible.
                Contains list of submodule's indices posing this problem.
                - `integer`: module is integer type but it's type is incompatible
                Contains list of submodule's indices posing this problem.

                Second level for shape:

                - `float`: module is floating point type and has incorrect shape
                - `integer`: module is integer type and has incorrect shape

                Third level for shape's `float` and `integer`:

                - `input`: module's input shape is incompatible with Tensor Cores
                Contains list of submodule's indices posing this problem.
                - `output`: module's output shape is incompatible with Tensor Cores
                Contains list of submodule's indices posing this problem.

                As it's hard to parse, it is suggested to use tips for readable output.
        """

        return self._analyse(module, "modules")

    def children(self, module: torch.nn.Module):
        r"""**Check Tensor Cores compatibility using** `children()` **method (shallow scanning).**

        Parameters
        ----------
        module : torch.nn.Module
                Module to be scanned for Tensor Cores compatibility

        Returns
        -------
        Nested dictionary
                Multilevel dictionary describing modules incompatible with tensor cores.
                First level consists of two fields:

                - `type`: incompatible type with TensorCores
                - `shape`: incompatible types with TensorCores

                Second level for type:

                - `float`: module is floating point type but it's type is incompatible.
                Contains list of submodule's indices posing this problem.
                - `integer`: module is integer type but it's type is incompatible
                Contains list of submodule's indices posing this problem.

                Second level for shape:

                - `float`: module is floating point type and has incorrect shape
                - `integer`: module is integer type and has incorrect shape

                Third level for shape's `float` and `integer`:

                - `input`: module's input shape is incompatible with Tensor Cores
                Contains list of submodule's indices posing this problem.
                - `output`: module's output shape is incompatible with Tensor Cores
                Contains list of submodule's indices posing this problem.

                As it's hard to parse, it is suggested to use tips for readable output.
        """

        return self._analyse(module, "children")

    def tips(self, module: torch.nn.Module) -> str:
        r"""**Return** `str` **representation of** `modules()` **method.**

        It is advised to use this function to get tips in order to easily fix
        possible performance issues related to Tensor Cores.

        Parameters
        ----------
        module : torch.nn.Module
                Module to be scanned

        Returns
        -------
        str
                String representing tips related to Tensor Cores.
        """
        data = self.modules(module)

        def types():
            _types = data["type"]

            def parse_type(is_float: bool, goal):
                key = "float" if is_float else "integer"
                if _types[key]:
                    return "\nModules where {} type is not {}:\n".format(
                        key, goal
                    ) + str(_types[key])
                return ""

            return parse_type(True, "torch.half") + parse_type(False, "torch.short")

        def shape():
            def parse_shape(dictionary, is_input: bool, goal):
                key = "inputs" if is_input else "outputs"
                if dictionary[key]:
                    return "\nModules where {} shape should be divisible by {}:\n".format(
                        key, goal
                    ) + str(
                        dictionary[key]
                    )
                return ""

            _shapes = data["shape"]

            def floating():
                _floats = _shapes["float"]
                return parse_shape(_floats, True, 8) + parse_shape(_floats, False, 8)

            def integer():
                _integers = _shapes["integer"]
                return parse_shape(_integers, True, 16) + parse_shape(
                    _integers, False, 16
                )

            return floating() + integer()

        output = types() + shape()
        if output != "":
            output = "TensorCores incompatible modules:" + output
        return output
