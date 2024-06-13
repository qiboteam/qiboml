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

    @abstractmethod
    def backward(self, input_grad: "ndarray"):
        pass

    @property
    def parameters(self) -> "ndarray":
        return self.circuit.get_parameters()

    @parameters.setter
    def parameters(self, x: list) -> None:
        self.circuit.set_parameters(x)


@dataclass
class SequentialQuantumModel(QuantumCircuitLayer):

    layers: list[QuantumCircuitLayer] = None

    def __post_init__(self):
        # in principle differently sized circuits might be passed
        if self.layers is None:
            self.layers = []
        nqubits = max([layer.circuit.nqubits for layer in self.layers])
        self.circuit = Circuit(nqubits)
        for layer in self.layers:
            circ = Circuit(nqubits)
            circ.add(layer.circuit.on_qubits(*range(layer.circuit.nqubits)))
            self.circuit = self.circuit + circ

    def _feed_input(self, x):
        self.layers[0]._feed_input(x)

    def backward(self, input_grad: "ndarray"):
        grad = input_grad
        for layer in self.layers:
            grad = layer.backward(grad)
        return grad