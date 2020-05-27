import torch

import torchfunc


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.convolution = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, 3),
            torch.nn.ReLU(inplace=True),  # Inplace may harm kernel fusion
            torch.nn.Conv2d(32, 128, 3, groups=32),  # Depthwise is slower in PyTorch
            torch.nn.ReLU(inplace=True),  # Same as before
            torch.nn.Conv2d(128, 250, 3),  # Wrong output size for TensorCores
        )

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(250, 64),  # Wrong input size for TensorCores
            torch.nn.ReLU(),  # Should be fine
            torch.nn.Linear(64, 10),  # Wrong output size for TensorCores
        )

    def forward(self, inputs):
        convolved = torch.nn.AdaptiveAvgPool2d(1)(self.convolution(inputs)).flatten()
        return self.classifier(convolved)


def test_report():
    goal = r"""
===========================GENERAL TIPS===========================

- Make sure you are running newest PyTorch version. See available releases: https://github.com/pytorch/pytorch/tags
- Use GPU for larger batches, CPU might be suitable for smaller jobs.
- Use mixed-precision training on GPU, preferably automated, e.g. NVIDIA Apex: https://github.com/NVIDIA/apex.

===========================SPECIFIC TIPS===========================

=======> Module should be an instance of torch.jit.ScriptModule.
See https://pytorch.org/docs/stable/jit.html for more information.
=======> NVIDIA's Apex is not installed. It is the easiest way to use mixed precision training.
See https://github.com/NVIDIA/apex for more information and installation.
=======> In-place operations might harm kernel fusion. Indices of those modules:
[3, 5]
You may want to remove inplace flag (see this issue: https://github.com/pytorch/pytorch/issues/23655)
=======> Depthwise convolutions are not currently using specialized kernel and might be slower.
See this issue: https://github.com/pytorch/pytorch/issues/18631 for more information.
Indices of those modules:
[4]
You may want to decrease number of groups (like it's done for ResNeXt) for possible speed & accuracy improvements.
=======> TensorCores incompatible modules:
Modules where float type is not torch.half:
[2, 4, 6, 8, 10]
Modules where inputs shape should be divisible by 8:
[2, 8]
Modules where outputs shape should be divisible by 8:
[6, 10]"""
    tips = torchfunc.performance.tips(Model())
    print(tips)
    assert tips == goal
