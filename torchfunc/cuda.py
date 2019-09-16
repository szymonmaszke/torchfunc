r"""
**This module provides CUDA related functionalities (e.g. resetting it's state).**

"""

import torch


def reset() -> None:
    r"""**Reset cuda state by emptying cache and collecting IPC.**

    Calls `torch.cuda.empty_cache()` and `torch.cuda.ipc_collect()` consecutively.

    Example::

        tensor = torch.cuda.FloatTensor(100, 100)
        del tensor
        torchfunc.cuda.reset() # Now memory is freed

    """
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
