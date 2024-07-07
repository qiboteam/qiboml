"""Some standard encoding and decoding layers"""

from dataclasses import dataclass
from typing import Union

from qibo import Circuit, gates
from qibo.config import raise_error

from qiboml.models.abstract import QuantumCircuitLayer


@dataclass
class QuantumEncodingLayer(QuantumCircuitLayer):

    def backward(self, input_grad: "ndarray") -> "ndarray":
        return input_grad


@dataclass
class BinaryEncodingLayer(QuantumEncodingLayer):

    def forward(self, x: "ndarray") -> Circuit:
        if x.shape[-1] != self.nqubits:
            raise_error(
                RuntimeError,
                f"Invalid input dimension {x.shape[-1]}, but nqubits is {self.nqubits}.",
            )
        circuit = self.circuit.copy()
        ones = x.ravel() == 1
        for bit in ones:
            circuit.add(gates.X(self.qubits[bit]))
        return circuit


@dataclass
class PhaseEncodingLayer(QuantumEncodingLayer):

    def __post_init__(self):
        super().__post_init__()
        self.circuit.add(gates.H(0))
        for q in self.qubits:
            if q != 0:
                self.circuit.add(gates.CNOT(0, q))
        for q in self.qubits:
            self.circuit.add(gates.RZ(q, theta=0.0))

    def forward(self, x: "ndarray") -> Circuit:
        self.circuit.set_parameters(x.ravel())
        return self.circuit


class AmplitudeEncodingLayer(QuantumEncodingLayer):
    pass


"""
   .
   .
   .
   .
"""


@dataclass
class QuantumDecodingLayer(QuantumCircuitLayer):

    nshots: int = 1000

    def __post_init__(self):
        super().__post_init__()
        self.circuit.add(gates.M(*self.qubits))

    def forward(self, x: Circuit) -> Circuit:
        return self.backend.execute_circuit(x + self.circuit, nshots=self.nshots)

    def backward(self):
        raise_error(NotImplementedError, "TO DO")


@dataclass
class ExpectationLayer(QuantumDecodingLayer):

    observable: Union["ndarray", "qibo.models.Hamiltonian"] = None

    def __post_init__(self):
        if self.observable is None:
            raise_error(
                RuntimeError,
                "Please provide an observable for expectation value calculation.",
            )
        super().__post_init__()

    def forward(self, x: Circuit) -> "ndarray":
        return self.observable.expectation_from_samples(
            super().forward(x).samples(),
            input_samples=True,
        )

    @property
    def output_shape(self):
        return (1,)


"""
   .
   .
   .
   .
"""
