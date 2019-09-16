****************
Related projects
****************

Below you can find other projects started by the same author and based on `PyTorch <https://pytorch.org/>`__ as well:

`torchdata: <https://github.com/szymonmaszke/torchdata>`__
==========================================================

**torchdata** extends `torch.utils.data.Dataset` and equips it with
functionalities known from `tensorflow.data <https://www.tensorflow.org/api_docs/python/tf/data/Dataset>`__
like `map` or `cache`.
All of that with minimal interference (single call to `super().__init__()`) with original
PyTorch's datasets.

Some functionalities:

* `torch.utils.data.IterableDataset` and `torch.utils.data.Dataset` support
* `map` or `apply` arbitrary functions to dataset
* `memory` or `disk` allows you to cache data (even partially, say `20%`)
* Concrete classes designed for file reading or database support

You can read documentation over at https://szymonmaszke.github.io/torchdata.

`torchnet: <https://github.com/szymonmaszke/torchnet>`__
==========================================================

**torchnet** tries to give you building blocks for your neural networks
and make working with them easier. In it you can find:

* Proven ImageNet SOTA convolutional building blocks (e.g. SqueezeNet, DenseNet)
* Parameter regularization as wrapper around cost functions
* Initialization pipeline for easier model setup
* Non-standard activation functions

You can read documentation over at https://szymonmaszke.github.io/torchnet.
