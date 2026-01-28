import importlib.metadata as im
from typing import Union

from numpy.typing import ArrayLike

from qiboml.backends.__init__ import MetaBackend

__version__ = im.version(__package__)

try:
    from tensorflow import Tensor as tf_tensor

    from qiboml.interfaces import keras

    ndarray = Union[ArrayLike, tf_tensor]
except ImportError:  # pragma: no cover
    pass

try:
    from torch import Tensor as pt_tensor

    from qiboml.interfaces import pytorch

    ndarray = Union[ArrayLike, pt_tensor]
except ImportError:  # pragma: no cover
    pass
