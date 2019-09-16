![torchfunc Logo](https://github.com/szymonmaszke/torchfunc/blob/master/assets/banner.png)

--------------------------------------------------------------------------------

| Version | Docs | Tests | Coverage | Style | PyPI | Python | PyTorch | Docker | Roadmap |
|---------------|-------|----------|-------|------|--------|---------|--------|---------|---------|
| [![Version](https://img.shields.io/static/v1?label=&message=0.1.0&color=377EF0&style=for-the-badge)](https://github.com/szymonmaszke/torchfunc/releases) | [![Documentation](https://img.shields.io/static/v1?label=&message=docs&color=EE4C2C&style=for-the-badge)](https://szymonmaszke.github.io/torchfunc/)  | ![Tests](https://github.com/szymonmaszke/torchfunc/workflows/test/badge.svg) | | | [![PyPI](https://img.shields.io/static/v1?label=&message=PyPI&color=377EF0&style=for-the-badge)](https://pypi.org/project/torchfunc/) | [![Python](https://img.shields.io/static/v1?label=&message=3.7&color=377EF0&style=for-the-badge&logo=python&logoColor=F8C63D)](https://www.python.org/) | [![PyTorch](https://img.shields.io/static/v1?label=&message=1.2.0&color=EE4C2C&style=for-the-badge)](https://pytorch.org/) | [![Docker](https://img.shields.io/static/v1?label=&message=docker&color=309cef&style=for-the-badge)](https://cloud.docker.com/u/szymonmaszke/repository/docker/szymonmaszke/torchfunc) | [![Roadmap](https://img.shields.io/static/v1?label=&message=roadmap&color=009688&style=for-the-badge)](https://github.com/szymonmaszke/torchfunc/blob/master/ROADMAP.md) |

[**torchfunc**](https://szymonmaszke.github.io/torchfunc/) is library revolving around [PyTorch](https://pytorch.org/) with a goal to help you with:

* Improving and analysing performance of your neural network
* Daily neural network duties (model size, seeding, performance measurements etc.)
* Plotting and visualizing modules
* Record neuron activity and tailor it to your specific task or target
* Get information about your host operating system, CUDA devices and others

# Quick examples

- Seed globaly, Freeze weights, check inference time and model size

```python
# Inb4 MNIST, you can use any module with those functions
model = torch.nn.Linear(784, 10)
frozen = torchfunc.module.freeze(model, bias=False)

with torchfunc.Timer() as timer:
  frozen(torch.randn(32, 784)
  print(timer.checkpoint()) # Time since the beginning
  frozen(torch.randn(128, 784)
  print(timer.checkpoint()) # Since last checkpoint
  
print(f"Overall time {timer}; Model size: {torchfunc.sizeof(frozen)}")
```

- Recorder and sum per-layer activation statistics as data passes through network:

```python
# MNIST classifier
model = torch.nn.Sequential(
    torch.nn.Linear(784, 100),
    torch.nn.ReLU(),
    torch.nn.Linear(100, 50),
    torch.nn.ReLU(),
    torch.nn.Linear(50, 10),
)
# Recorder which sums layer inputs from consecutive forward calls
recorder = torchfunc.record.ForwardPreRecorder(reduction=lambda x, y: x+y)
# Record inputs going into Linear(100, 50) and Linear(50, 10)
recorder.children(model, indices=(2, 3))
# Train your network normally (or pass data through it)
...
# Save tensors (of shape 100 and 50) in folder, each named 1.pt and 2.pt respectively
recorder.save(pathlib.Path("./analysis"))
```

For performance tips, plotting and other check [**torchfunc documentation**](https://szymonmaszke.github.io/torchfunc/).

# Installation

## [pip](<https://pypi.org/project/torchfunc/>)

### Latest release:

```shell
pip install --user torchfunc
```

### Nightly:

```shell
pip install --user torchfunc-nightly
```

## [Docker](https://cloud.docker.com/repository/docker/szymonmaszke/torchfunc)

__CPU standalone__ and various versions of __GPU enabled__ images are available
at [dockerhub](https://cloud.docker.com/repository/docker/szymonmaszke/torchfunc).

For CPU quickstart, issue:

```shell  
docker pull szymonmaszke/torchfunc:18.04
```

Nightly builds are also available, just prefix tag with `nightly_`. If you are going for `GPU` image make sure you have
[nvidia/docker](https://github.com/NVIDIA/nvidia-docker) installed and it's runtime set.

# Contributing

If you find any issue or you think some functionality may be useful to others and fits this library, please [open new Issue](https://help.github.com/en/articles/creating-an-issue) or [create Pull Request](https://help.github.com/en/articles/creating-a-pull-request-from-a-fork).

To get an overview of something which one can done to help this project, see [Roadmap](https://github.com/szymonmaszke/torchfunc/blob/master/ROADMAP.md)
