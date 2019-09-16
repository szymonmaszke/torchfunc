r"""

**This package allows you to measure and improve performance of your neural networks.**

Following functions should be considered as general recommendations.
For specific info/tips, use specific module.


"""

import typing

import torch

import torchfunc

from . import layers, technology
from .layers import Depthwise, Inplace
from .technology import TensorCores


def report(module: torch.nn.Module) -> typing.Dict[str, typing.Any]:
    r"""**Run essential module's performance analysis with default settings.**

    Following tests will be performed:

    - Module being an instance of `torch.nn.ScriptModule`
    - Apex (mixed precision training) availability
    - Any inplace ops used
    - Analysis of compliance with `TensorCores` technology
    - Any depthwise convolution used

    Report returns data in machine-ready type; if you wish to have easy to follow
    guidelines use function `tips`.

    **Example**::

        model = torch.nn.Sequential(
            torch.nn.Linear(784, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 50),
            torch.nn.ReLU(),
            torch.nn.Linear(50, 10),
        )
        report = torchfunc.performance.report(model)

    Parameters
    ----------
    module: torch.nn.Module
            Module to be tested against test suite.

    Returns
    -------
    Dict[str, Any]
            Dictionary with keys:

            - torchscript: True if module is an instance of torch.jit.ScriptModule
            - apex: True if apex installed (recommended mixed precision training library from NVidia for PyTorch)
            - tensorcores: same as `torchscript.performance.technology.TensorCores`
            - inplace: same as `torchscript.performance.layers.Inplace`
            - depthwise: same as `torchscript.performance.layers.Depthwise`

    """

    return {
        "torchscript": isinstance(module, torch.jit.ScriptModule),
        "apex": torchfunc.installed("apex"),
        "tensorcores": TensorCores().modules(module),
        "inplace": Inplace().modules(module),
        "depthwise": Depthwise().modules(module),
    }


# Parse TensorCores
def tips(module: torch.nn.Module, general: bool = True, specific: bool = True):
    r"""**Return string describing possible performance improvements one can undertake.**

    Internally report will be called and it's output parsed and described.
    It is the easiest way to get information about your module/network
    and to quickly check possible performance improvements you could use.

    **Example**::

        model = torch.nn.Sequential(
            torch.nn.Linear(784, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 50),
            torch.nn.ReLU(),
            torch.nn.Linear(50, 10),
        )
        print(torchfunc.performance.tips(model)) # Display found vulnerabilities

    Parameters
    ----------
    module: torch.nn.Module
            Module to be tested against test suite.
    general: bool, optional
            Return general (not specific to your module) tips. Default: True
    specific: bool, optional
            Return specific tips for your module. Default: True

    Returns
    -------
    str
            Human readable version of report, highlighting steps
            one can take to improve module's performance.

    """
    data = report(module)
    general_tips = (
        "\n===========================GENERAL TIPS===========================\n"
        + "\n- Make sure you are running newest PyTorch version. "
        + "See available releases: https://github.com/pytorch/pytorch/tags\n"
        + "- Use GPU for larger batches, CPU might be suitable for smaller jobs.\n"
        + "- Use mixed-precision training on GPU, preferably automated, e.g. NVIDIA Apex: https://github.com/NVIDIA/apex.\n"
    )
    specific_tips = (
        "\n===========================SPECIFIC TIPS===========================\n"
    )
    if not data["torchscript"]:
        specific_tips += (
            "\n- Module is not an instance of torch.jit.ScriptModule.\n"
            + "See https://pytorch.org/docs/stable/jit.html for more information."
        )
    if not data["apex"]:
        specific_tips += (
            "\n- NVIDIA's Apex is not installed. It is the easiest way to use mixed precision training.\n"
            + "See https://github.com/NVIDIA/apex for more information."
        )

    inplace = tuple(data["inplace"])
    if inplace:
        specific_tips += "\n- Some operations are in-place. It might harm kernel fusion. Indices of those modules:\n"
        specific_tips += str(list(inplace))
        specific_tips += "\nYou may want to remove inplace flag (as of torch 1.2.0)"

    depthwise = tuple(data["depthwise"])
    if depthwise:
        specific_tips += "\n- Some layers are depthwise convolutions. Those ARE NOT using specialized kernel and might be slower.\n"
        specific_tips += "Indices of those modules:\n"
        specific_tips += str(list(depthwise))
        specific_tips += "\nYou may want to decrease number of groups (like it's done for ResNeXt) for possible speed & accuracy improvements."

    results = ""
    if general:
        results += general_tips
    if specific:
        results += specific_tips

    return results


# Add automatic model's improvements (fix data type)
# def improve(module: torch.nn.Module):
#     pass
