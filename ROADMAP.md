# Roadmap

This document highlights possible improvements and additional implementations
driving this library towards next release (version `0.2.0`).

You can check whether someone started work on this task in issues under the label
`roadmap` and specific `improvement` or `implementation`.

Any help with tasks described below will be highly appreciated, thanks!

## Improvements

Following tasks might be considered improvements over what is already done.
Please use label `roadmap` and `improvement` and describe the task you are willing to take
(or describe new task you would consider an improvement).

- Increase test coverage (especially `torchfunc.performance` and `torchfunc.record` packages)
- Testing with [Hypothesis](https://github.com/HypothesisWorks/hypothesis)
- Documentation descriptions
- Tutorials creation
- Conda release

## Implementations

- `torchfunc.performance.technology` extensions, namely:
  - Approximation of [Arithmetic Intensity](https://docs.nvidia.com/deeplearning/sdk/dl-performance-guide/index.html#math-mem) for module
  - Approximation of good batch sizes for specific modules to avoid wave quantization (see [here](https://docs.nvidia.com/deeplearning/sdk/dl-performance-guide/index.html#gpu-execution))
  - Configurable invasive automatic improvement of module (e.g. changing data types to be tensor cores compatible), could be easily divided into multiple tasks.
  Open a PR if you are willing to discuss and/or talk about this functionality.

- `torchfunc.plot` extensions, namely:
  - Readable Visualization of Convolutional layers activation
  - Visualization of RNN activations
  - Basic visualization of Linear layer connections and their values
