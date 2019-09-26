import sys
import time

import torch

import torchfunc


def test_timer_context_manager():
    with torchfunc.Timer() as timer:
        time.sleep(1)
        last_in_block = timer.checkpoint()  # register checkpoint
    last_time = timer.checkpoint()
    time.sleep(1)
    assert last_time == timer.checkpoint() == timer.time()
    assert last_in_block != last_time


def test_timer_decorator():
    @torchfunc.Timer()
    def wrapped():
        time.sleep(1)
        result = 0
        for value in range(11):
            result += value
        return int(result / 55)

    value, passed_time = wrapped()
    assert value == 1
    assert passed_time > 1


def test_seed():
    torchfunc.seed(0)
    assert 0 == torch.initial_seed()


def test_seed_str():
    assert str(torchfunc.seed(0)) == "torchfunc.seed"


def test_seed_representation():
    assert repr(torchfunc.seed(0)) == "torchfunc.seed(value=0, cuda=False)"


def test_seed_context_manager():
    first_seed = torch.initial_seed()
    with torchfunc.seed(0):
        assert 0 == torch.initial_seed()
    assert torch.initial_seed() == first_seed


def test_seed_decorator():
    first_seed = torch.initial_seed()

    @torchfunc.seed(0)
    def wrapped():
        assert 0 == torch.initial_seed()

    wrapped()
    assert torch.initial_seed() == first_seed


def test_info():
    assert isinstance(torchfunc.info(), str)


def test_sizeof_tensor():
    assert torchfunc.sizeof(torch.FloatTensor(12, 12)) == 12 * 12 * 4


def test_sizeof_model():
    model = torch.nn.Linear(20, 20)
    bias = 20 * 4
    weights = 20 * 20 * 4
    assert torchfunc.sizeof(model) == bias + weights
