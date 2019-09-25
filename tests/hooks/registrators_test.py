import pathlib
import shutil
import tempfile

import torch

import pytest
import torchfunc


@pytest.fixture
def registrator():
    class Hook:
        def __init__(self):
            self.counter = 0

        def __call__(self, *_):
            self.counter += 1

        def __int__(self):
            return self.counter

    model = torch.nn.Sequential(
        torch.nn.Linear(784, 100),
        torch.nn.ReLU(),
        torch.nn.Linear(100, 50),
        torch.nn.ReLU(),
        torch.nn.Linear(50, 10),
    )

    _registrator = torchfunc.hooks.registrators.Forward(Hook())
    _registrator.modules(model, types=(torch.nn.Linear,))

    for _ in range(1000):
        model(torch.randn(1, 784))

    return _registrator


def test_hook_calls(registrator):
    assert int(registrator.hook) == 3 * 1000


def test_len(registrator):
    assert len(registrator) == 3


def test_remove(registrator):
    registrator.remove(0)
    assert len(registrator) == 2
