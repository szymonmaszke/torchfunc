import pathlib
import shutil
import tempfile

import torch

import pytest
import torchfunc
import torchvision


@pytest.fixture
def recorded():
    model = torch.nn.Sequential(
        torch.nn.Linear(784, 100),
        torch.nn.ReLU(),
        torch.nn.Linear(100, 50),
        torch.nn.ReLU(),
        torch.nn.Linear(50, 10),
    )

    recorder = torchfunc.hooks.recorders.ForwardPre(reduction=lambda x, y: x + y)
    recorder.children(model, indices=(2, 3))

    batch_size = 64
    # Create tempdir for data storage in torchfunc?
    for data, _ in torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            tempfile.gettempdir(),
            train=False,
            download=True,
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Lambda(lambda data: data.flatten()),
                ]
            ),
        ),
        batch_size=batch_size,
        drop_last=True,
    ):
        model(data)

    return batch_size, recorder


def test_len(recorded):
    batch_size, recorder = recorded
    assert len(recorder) == 2


def test_shapes(recorded):
    batch_size, recorder = recorded
    assert recorder[0].shape == (batch_size, 100)
    assert recorder[1].shape == (batch_size, 50)


def test_samples(recorded):
    batch_size, recorder = recorded
    assert recorder.samples(0) == 10000 // batch_size
    for sample in recorder.iter_samples():
        assert sample == 10000 // batch_size


def test_save(recorded):
    batch_size, recorder = recorded
    folder = "TORCHFUNC_SAVED_DATA"
    temp_dir = pathlib.Path(tempfile.gettempdir()) / folder
    recorder.save(temp_dir, mkdir=True)
    assert len([file for file in temp_dir.iterdir()]) == 2
    shutil.rmtree(temp_dir)
