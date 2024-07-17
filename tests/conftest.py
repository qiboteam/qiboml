"""conftest.py.

Pytest fixtures.
"""

import pytest
from qibo.backends import construct_backend

# backends to be tested
BACKENDS = [
    "numpy",
    "tensorflow",
    "pytorch",
    "qibojit-numba",
    "qibojit-cupy",
    "qibojit-cuquantum",
]
