"""Defines the general structure of a qiboml module"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from qibo import Circuit

from qiboml.backends import JaxBackend


@dataclass
class QuantumCircuitLayer(ABC):

    circuit: Circuit = None
    backend: "qibo.backends.Backend" = JaxBackend()

    @abstractmethod
    def __post_init__(self):
        pass

    @abstractmethod
    def _feed_input(self, x):
        pass

    def forward(self, x):
        """Performs the forward pass: prepares the input and execute the circuit."""
        self._feed_input(x)
        return self.backend.execute_circuit(self.circuit)

    @abstractmethod
    def backward(self, input_grad: "ndarray"):
        pass

    @property
    def parameters(self):
        return self.circuit.get_parameters()

    @parameters.setter
    def parameters(self, x: "ndarray"):
        self.circuit.set_parameters(x)


@dataclass
class SequentialQuantumModel(QuantumCircuitLayer):

    layers: list[QuantumCircuitLayer] = []

    def __post_init__(self):
        # in principle differently sized circuits might be passed
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
