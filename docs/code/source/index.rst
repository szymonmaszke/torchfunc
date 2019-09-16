:github_url: https://github.com/szymonmaszke/torchfunc

*********
torchfunc
*********

**torchfunc** is PyTorch oriented library with a goal to help you with:

* Improving and analysing performance of your neural network
* Plotting and visualizing modules
* Record neuron activity and tailor it to your specific task or target
* Get information about your host operating system, CUDA devices and others
* Day-to-day neural network related duties (model size, seeding, performance measurements etc.)

See `related projects <https://szymonmaszke.github.io/torchfunc/related.html>`__ for `td.data`-like
datasets in PyTorch and soon others as well!

Modules
#######

.. toctree::
   :glob:
   :maxdepth: 1

   packages/*
   
.. toctree::
   :hidden:

   related


Installation
############

Following installation methods are available:

`pip: <https://pypi.org/project/torchfunc/>`__
==============================================

To install latest release:

.. code-block:: shell

  pip install --user torchfunc

To install `nightly` version:

.. code-block:: shell

  pip install --user torchfunc-nightly


`Docker: <https://cloud.docker.com/repository/docker/szymonmaszke/torchfunc>`__
===============================================================================

Various `torchfunc` images are available both CPU and GPU-enabled.
You can find them in Docker Cloud at `szymonmaszke/torchfunc`

CPU
---

CPU image is based on `ubuntu:18.04 <https://hub.docker.com/_/ubuntu>`__ and
official release can be pulled with:

.. code-block:: shell
  
  docker pull szymonmaszke/torchfunc:18.04

For `nightly` release:

.. code-block:: shell
  
  docker pull szymonmaszke/torchfunc:nightly_18.04

This image is significantly lighter due to lack of GPU support.

GPU
---

All images are based on `nvidia/cuda <https://hub.docker.com/r/nvidia/cuda/>`__ Docker image.
Each has corresponding CUDA version tag ( `10.1`, `10` and `9.2`) CUDNN7 support
and base image ( `ubuntu:18.04 <https://hub.docker.com/_/ubuntu>`__ ).

Following images are available:

- `10.1-cudnn7-runtime-ubuntu18.04`
- `10.1-runtime-ubuntu18.04`
- `10.0-cudnn7-runtime-ubuntu18.04`
- `10.0-runtime-ubuntu18.04`
- `9.2-cudnn7-runtime-ubuntu18.04`
- `9.2-runtime-ubuntu18.04`

Example pull:

.. code-block:: shell
  
  docker pull szymonmaszke/torchfunc:10.1-cudnn7-runtime-ubuntu18.04

You can use `nightly` builds as well, just prefix the tag with `nightly_`, for example
like this:

.. code-block:: shell
  
  docker pull szymonmaszke/torchfunc:nightly_10.1-cudnn7-runtime-ubuntu18.04


`conda: <https://anaconda.org/conda-forge/torchfunc>`__
=======================================================

**TO BE ADDED**

.. code-block:: shell

  conda install -c conda-forge torchfunc
