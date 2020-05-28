<img align="left" width="256" height="256" src="https://github.com/szymonmaszke/torchfunc/blob/master/assets/logos/medium.png">

* Improve and analyse performance of your neural network (e.g. Tensor Cores compatibility)
* Record/analyse internal state of `torch.nn.Module` as data passes through it
* Do the above based on external conditions (using single `Callable` to specify it)
* Day-to-day neural network related duties (model size, seeding, time measurements etc.)
* Get information about your host operating system, `torch.nn.Module` device, CUDA
capabilities etc.


| Version | Docs | Tests | Coverage | Style | PyPI | Python | PyTorch | Docker | Roadmap |
|---------|------|-------|----------|-------|------|--------|---------|--------|---------|
| [![Version](https://img.shields.io/static/v1?label=&message=0.2.0&color=377EF0&style=for-the-badge)](https://github.com/szymonmaszke/torchfunc/releases) | [![Documentation](https://img.shields.io/static/v1?label=&message=docs&color=EE4C2C&style=for-the-badge)](https://szymonmaszke.github.io/torchfunc/)  | ![Tests](https://github.com/szymonmaszke/torchfunc/workflows/test/badge.svg) | ![Coverage](https://img.shields.io/codecov/c/github/szymonmaszke/torchfunc?label=%20&logo=codecov&style=for-the-badge) | [![codebeat](https://img.shields.io/static/v1?label=&message=CB&color=27A8E0&style=for-the-badge)](https://codebeat.co/projects/github-com-szymonmaszke-torchfunc-master) | [![PyPI](https://img.shields.io/static/v1?label=&message=PyPI&color=377EF0&style=for-the-badge)](https://pypi.org/project/torchfunc/) | [![Python](https://img.shields.io/static/v1?label=&message=3.6&color=377EF0&style=for-the-badge&logo=python&logoColor=F8C63D)](https://www.python.org/) | [![PyTorch](https://img.shields.io/static/v1?label=&message=>=1.2.0&color=EE4C2C&style=for-the-badge)](https://pytorch.org/) | [![Docker](https://img.shields.io/static/v1?label=&message=docker&color=309cef&style=for-the-badge)](https://hub.docker.com/r/szymonmaszke/torchfunc) | [![Roadmap](https://img.shields.io/static/v1?label=&message=roadmap&color=009688&style=for-the-badge)](https://github.com/szymonmaszke/torchfunc/blob/master/ROADMAP.md) |

# :bulb: Examples

__Check documentation here:__ [https://szymonmaszke.github.io/torchfunc](https://szymonmaszke.github.io/torchfunc)

## 1. Getting performance tips

- __Get instant performance tips about your module. All problems described by comments
will be shown by `torchfunc.performance.tips`:__

```python
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
            torch.nn.ReLU(),  # Fine, no info about this layer
            torch.nn.Linear(64, 10),  # Wrong output size for TensorCores
        )

    def forward(self, inputs):
        convolved = torch.nn.AdaptiveAvgPool2d(1)(self.convolution(inputs)).flatten()
        return self.classifier(convolved)

# All you have to do
print(torchfunc.performance.tips(Model()))
```

## 2. Seeding, weight freezing and others

- __Seed globaly (including `numpy` and `cuda`), freeze weights, check inference time and model size:__

```python
# Inb4 MNIST, you can use any module with those functions
model = torch.nn.Linear(784, 10)
torchfunc.seed(0)
frozen = torchfunc.module.freeze(model, bias=False)

with torchfunc.Timer() as timer:
  frozen(torch.randn(32, 784)
  print(timer.checkpoint()) # Time since the beginning
  frozen(torch.randn(128, 784)
  print(timer.checkpoint()) # Since last checkpoint

print(f"Overall time {timer}; Model size: {torchfunc.sizeof(frozen)}")
```

## 3. Record `torch.nn.Module` internal state

- __Record and sum per-layer activation statistics as data passes through network:__

```python
# Still MNIST but any module can be put in it's place
model = torch.nn.Sequential(
    torch.nn.Linear(784, 100),
    torch.nn.ReLU(),
    torch.nn.Linear(100, 50),
    torch.nn.ReLU(),
    torch.nn.Linear(50, 10),
)
# Recorder which sums all inputs to layers
recorder = torchfunc.hooks.recorders.ForwardPre(reduction=lambda x, y: x+y)
# Record only for torch.nn.Linear
recorder.children(model, types=(torch.nn.Linear,))
# Train your network normally (or pass data through it)
...
# Activations of all neurons of first layer!
print(recorder[1]) # You can also post-process this data easily with apply
```

For other examples (and how to use condition), see [documentation](https://szymonmaszke.github.io/torchfunc/)

# :wrench: Installation

## :snake: [pip](<https://pypi.org/project/torchfunc/>)

### Latest release:

```shell
pip install --user torchfunc
```

### Nightly:

```shell
pip install --user torchfunc-nightly
```

## :whale2: [Docker](https://hub.docker.com/r/szymonmaszke/torchfunc)

__CPU standalone__ and various versions of __GPU enabled__ images are available
at [dockerhub](https://hub.docker.com/r/szymonmaszke/torchfunc/tags).

For CPU quickstart, issue:

```shell
docker pull szymonmaszke/torchfunc:18.04
```

Nightly builds are also available, just prefix tag with `nightly_`. If you are going for `GPU` image make sure you have
[nvidia/docker](https://github.com/NVIDIA/nvidia-docker) installed and it's runtime set.

# :question: Contributing

If you find any issue or you think some functionality may be useful to others and fits this library, please [open new Issue](https://help.github.com/en/articles/creating-an-issue) or [create Pull Request](https://help.github.com/en/articles/creating-a-pull-request-from-a-fork).

To get an overview of things one can do to help this project, see [Roadmap](https://github.com/szymonmaszke/torchfunc/blob/master/ROADMAP.md).
