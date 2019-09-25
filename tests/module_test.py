import itertools
import pathlib
import shutil
import tempfile

import torch

import pytest
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
        assert _test_value(bias if "bias" in name else weights, parameter)


def test_freezing():
    for model, bias, weights in itertools.product([True, False], repeat=3):
        _test_network(model, bias, weights)


@pytest.fixture
def snapshot():
    modules = [
        torch.nn.Sequential(
            torch.nn.Linear(100, 50), torch.nn.Linear(50, 50), torch.nn.Linear(50, 10)
        )
        for _ in range(4)
    ]

    snapshot = torchfunc.module.Snapshot()
    snapshot += modules[0]
    snapshot += modules[1]
    return snapshot


def test_snapshot_len(snapshot):
    assert len(snapshot) == 2


def test_snapshot_save(snapshot):
    folder = "TORCHFUNC_SNAPSHOT"
    temp_dir = pathlib.Path(tempfile.gettempdir()) / folder
    temp_dir.mkdir()
    snapshot.save(temp_dir)
    assert len([file for file in temp_dir.iterdir()]) == 2
    shutil.rmtree(temp_dir)
