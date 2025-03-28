import importlib.metadata as im
from typing import Union

import numpy.typing as npt

from qiboml.backends.__init__ import MetaBackend

__version__ = im.version(__package__)

ndarray = npt.NDArray

try:
    from tensorflow import Tensor as tf_tensor

    from qiboml.interfaces import keras

    ndarray = Union[ndarray, tf_tensor]
except ImportError:  # pragma: no cover
    pass

try:
    from torch import Tensor as pt_tensor

    from qiboml.interfaces import pytorch

    ndarray = Union[ndarray, pt_tensor]
except ImportError:  # pragma: no cover
    pass
