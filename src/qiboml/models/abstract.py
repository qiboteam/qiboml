"""Defines the general structure of a qiboml module"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from qibo import Circuit
from qibo.config import raise_error
from qibo.gates import abstract

from qiboml.backends import TensorflowBackend as JaxBackend


@dataclass
class QuantumCircuitLayer(ABC):

    nqubits: int
    qubits: list[int] = None
    circuit: Circuit = None
    initial_state: "ndarray" = None
    backend: "qibo.backends.Backend" = JaxBackend()

    def __post_init__(self) -> None:
        if self.qubits is None:
            self.qubits = list(range(self.nqubits))
        self.circuit = Circuit(self.nqubits)

    @abstractmethod
    def forward(self, x):
        pass

    def __call__(self, x):
        return self.forward(x)

    @property
    def parameters(self) -> "ndarray":
        return self.backend.cast(self.circuit.get_parameters())

    @parameters.setter
    def parameters(self, x: "ndarray") -> None:
        self.circuit.set_parameters(x)
