import importlib.metadata as im

from qiboml.backends.__init__ import MetaBackend
from qiboml.models import keras, pytorch

__version__ = im.version(__package__)
