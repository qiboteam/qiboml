import importlib.metadata as im
from typing import Union

import numpy.typing as npt

from qiboml.backends.__init__ import MetaBackend

__version__ = im.version(__package__)

ndarray = npt.NDArray


def __getattr__(name: str):
    if name == "keras":
        try:
            from importlib import import_module

            return import_module("qiboml.interfaces.keras")
        except ImportError:
            return None
    if name == "pytorch":
        try:
            from importlib import import_module

            return import_module("qiboml.interfaces.pytorch")
        except ImportError:
            return None
    raise AttributeError(f"module {__name__} has no attribute {name}")
