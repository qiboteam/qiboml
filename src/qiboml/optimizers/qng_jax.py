"""Module with the implementation of the quantum natural gradient."""

from typing import Optional

from jax import grad
from qibo import Circuit
from qibo.backends import Backend, _check_backend
from qibo.quantum_info.metrics import quantum_fisher_information_matrix

from qiboml.backends.jax import JaxBackend


class QuantumNaturalGradientJax:
    def __init__(
        self,
        circuit: Circuit,
        learning_rate: Optional[float] = 1e-3,
        backend: Optional[Backend] = None,
        **kwargs,
    ):
        self.backend = _check_backend(backend)
        assert isinstance(self.backend, JaxBackend)
