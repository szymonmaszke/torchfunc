r"""
**This package allows you to get info and tips about performance of your neural networks.**

Following functions should be considered as general recommendations.
For specific/customized tips, use specific submodules.

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


# Text parsing is not the prettiest thing ever :(
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

    def general_tips():
        return (
            "\n===========================GENERAL TIPS===========================\n"
            + "\n- Make sure you are running newest PyTorch version. "
            + "See available releases: https://github.com/pytorch/pytorch/tags\n"
            + "- Use GPU for larger batches, CPU might be suitable for smaller jobs.\n"
            + "- Use mixed-precision training on GPU, preferably automated, e.g. NVIDIA Apex: https://github.com/NVIDIA/apex.\n"
        )

    def specific_tips():
        def parse_string(text: str) -> str:
            if text != "":
                return "\n=======> " + text
            return text

        def torchscript():
            if not isinstance(module, torch.jit.ScriptModule):
                return (
                    "Module should be an instance of torch.jit.ScriptModule.\n"
                    + "See https://pytorch.org/docs/stable/jit.html for more information."
                )
            return ""

        def apex():
            if not torchfunc.installed("apex"):
                return (
                    "NVIDIA's Apex is not installed. It is the easiest way to use mixed precision training.\n"
                    + "See https://github.com/NVIDIA/apex for more information and installation."
                )
            return ""

        specific_tips = ""
        specific_tips += parse_string(torchscript())
        specific_tips += parse_string(apex())
        specific_tips += parse_string(Inplace().tips(module))
        specific_tips += parse_string(Depthwise().tips(module))
        specific_tips += parse_string(TensorCores().tips(module))
        if specific_tips != "":
            specific_tips = (
                "\n===========================SPECIFIC TIPS===========================\n"
                + specific_tips
            )
        return specific_tips

    ###########################################################################
    #
    #                               MAIN LOGIC
    #
    ###########################################################################

    results = ""
    if general:
        results += general_tips()
    if specific:
        results += specific_tips()

    return results
