import torch

import torchfunc


def test_reset():
    if torch.cuda.is_available():
        tensor = torch.cuda.FloatTensor(100, 100)
        torchfunc.sizeof(tensor)
        cached = torch.cuda.max_memory_cached()
        del tensor
        torch.cuda.reset_max_memory_cached()
        assert cached == torch.cuda.max_memory_cached()
        torchfunc.cuda.reset()
        torch.cuda.reset_max_memory_cached()
        assert 0 == torch.cuda.max_memory_cached()
