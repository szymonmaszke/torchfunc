import itertools

import torch

import torchfunc


def _test_value(condition, parameter):
    if condition:
        return not parameter.requires_grad
    return parameter.requires_grad


def _test_network(model: bool, bias: bool, weights: bool):
    module = (
        torch.nn.Sequential(
            torch.nn.Linear(100, 50), torch.nn.Linear(50, 50), torch.nn.Linear(50, 10)
        )
        if model
        else torch.nn.Linear(20, 40)
    )

    for parameter in module.parameters():
        assert parameter.requires_grad
    for name, parameter in torchfunc.module.freeze(
        module, weights, bias
    ).named_parameters():
        print(name)
        assert _test_value(bias if "bias" in name else weights, parameter)


def test_freezing():
    for model, bias, weights in itertools.product([True, False], repeat=3):
        print(model, bias, weights)
        _test_network(model, bias, weights)
