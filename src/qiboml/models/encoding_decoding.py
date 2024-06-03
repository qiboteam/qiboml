"""Some standard encoding and decoding layers"""

from dataclasses import dataclass
from typing import Union

from qibo import Circuit, gates
from qibo.config import raise_error

from qiboml.models.abstract import QuantumCircuitLayer


@dataclass
class QuantumEncodingLayer(QuantumCircuitLayer):
    pass


@dataclass
class BinaryEncodingLayer(QuantumEncodingLayer):

    def forward(self, x: "ndarray") -> Circuit:
        if isinstance(x, Circuit):
            raise_error(RuntimeError, "Passed a `Circuit` as input data.")
        if x.shape[-1] != self.nqubits:
            raise_error(
                RuntimeError,
                f"Invalid input dimension {x.shape[-1]}, but nqubits is {self.nqubits}.",
            )
        circuit = self.circuit.copy()
        for q, bit in zip(self.qubits, x):
            if bit:
                circuit.add(gates.X(q))
        return circuit

    def backward(self, input_grad: "ndarray") -> "ndarray":
        return input_grad


class PhaseEncodingLayer(QuantumEncodingLayer):
    pass


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

    def forward(self, x: Circuit) -> "ndarray":
        return self.observable.expectation_from_samples(
            super().forward(x).frequencies()
        )


"""
   .
   .
   .
   .
"""
