r"""
**This package provides hook related functionalities (e.g. recording network state,
easier hook registration).**

To record neural network states and how it interacts with data, see module `recorders`.
To register different `hook` for your neural network (e.g. concatenated modules),
see module `registrator`.

"""

from . import recorders, registrators
