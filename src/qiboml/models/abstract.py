"""Defines the general structure of a qiboml module"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generator

from qibo import Circuit
from qibo.backends import Backend, GlobalBackend
from qibo.config import raise_error
from qibo.gates import abstract

from qiboml import ndarray
from qiboml.backends import JaxBackend


@dataclass
class QuantumCircuitLayer(ABC):

    nqubits: int
    qubits: list[int] = None
    _circuit: Circuit = None
    backend: Backend = GlobalBackend()

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
    def has_parameters(self):
        if len(list(self.parameters)) > 0:
            return True
        return False

    @property
    def parameters(self) -> Generator[ndarray, None, None]:
        return (gate.parameters for gate in self.circuit.trainable_gates)

    @parameters.setter
    def parameters(self, params: list[ndarray]):
        for param, gate in zip(params, self.circuit.trainable_gates):
            gate.parameters = [param]

    @property
    def circuit(self) -> Circuit:
        return self._circuit


def _run_layers(x: ndarray, layers: list[QuantumCircuitLayer], parameters):
    index = 0
    for layer in layers:
        if layer.has_parameters:
            layer.parameters = parameters[index]
            index += 1
        x = layer.forward(x)
    return x
