import torch

import torchfunc


def test_depthwise():
    model = torch.nn.Sequential(
        torch.nn.Conv1d(64, 64, kernel_size=3, groups=64),
        torch.nn.Conv2d(3, 32, kernel_size=3, groups=1),
        torch.nn.Conv2d(32, 32, kernel_size=3, groups=32),
    )
    values = tuple(torchfunc.performance.layers.Depthwise().modules(model))
    assert values == (1, 3)


def test_inplace():
    model = torch.nn.Sequential(
        torch.nn.ReLU(inplace=True),
        torch.nn.ReLU6(inplace=True),
        torch.nn.Conv1d(64, 64, kernel_size=3, groups=64),
        torch.nn.Conv2d(3, 32, kernel_size=3, groups=1),
        torch.nn.Conv2d(32, 32, kernel_size=3, groups=32),
    )
    values = torchfunc.performance.layers.Inplace().children(model)
    for i, value in enumerate(values):
        assert i == value
