import pathlib
import shutil
import tempfile

import torch

import pytest
import torchfunc


@pytest.fixture
def recorder():
    model = torch.nn.Sequential(
        torch.nn.Linear(784, 100),
        torch.nn.ReLU(),
        torch.nn.Linear(100, 50),
        torch.nn.ReLU(),
        torch.nn.Linear(50, 10),
    )

    _recorder = torchfunc.hooks.recorders.ForwardPre(reduction=lambda x, y: x + y)
    _recorder.children(model, indices=(2, 3))

    for _ in range(1000):
        model(torch.randn(1, 784))

    return _recorder


def test_len(recorder):
    assert len(recorder) == 2


def test_shapes(recorder):
    assert recorder[0].shape == (1, 100)
    assert recorder[1].shape == (1, 50)


def test_samples(recorder):
    assert recorder.samples(0) == 1000
    for sample in recorder.iter_samples():
        assert sample == 1000


def test_save(recorder):
    folder = "TORCHFUNC_RECORDER"
    temp_dir = pathlib.Path(tempfile.gettempdir()) / folder
    recorder.save(temp_dir, mkdir=True)
    assert len([file for file in temp_dir.iterdir()]) == 2
    shutil.rmtree(temp_dir)


def test_remove(recorder):
    recorder.remove(0)
    assert len(recorder) == 1
