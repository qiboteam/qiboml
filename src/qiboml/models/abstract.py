"""Defines the general structure of a qiboml module"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from qibo import Circuit
from qibo.backends import Backend
from qibo.config import raise_error
from qibo.gates import abstract

from qiboml import ndarray
from qiboml.backends import JaxBackend


@dataclass
class QuantumCircuitLayer(ABC):

    nqubits: int
    qubits: list[int] = None
    _circuit: Circuit = None
    backend: Backend = JaxBackend()

    def __post_init__(self) -> None:
        if self.qubits is None:
            self.qubits = list(range(self.nqubits))
        self._circuit = Circuit(self.nqubits)

    @abstractmethod
    def forward(self, x):  # pragma: no cover
        pass

    def __call__(self, x):
        return self.forward(x)

    @property
    def parameters(self) -> ndarray:
        return self.backend.cast(self.circuit.get_parameters())

    @parameters.setter
    def parameters(self, params: ndarray):
        self._circuit.set_parameters(self.backend.cast(params.ravel()))

    @property
    def circuit(self) -> Circuit:
        return self._circuit


def _run_layers(x: ndarray, layers: list[QuantumCircuitLayer]):
    for layer in layers:
        x = layer.forward(x)
    return x
