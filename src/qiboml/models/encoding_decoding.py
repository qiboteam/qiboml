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

    def _feed_input(self, x: "ndarray"):
        if x.shape[-1] != self.nqubits:
            raise_error(
                RuntimeError,
                f"Invalid input dimension {x.shape[-1]}, but nqubits is {self.nqubits}.",
            )
        self.circuit = Circuit(self.nqubits)
        for i, bit in enumerate(x):
            if bit:
                self.circuit.add(gates.X(i))

    def backward(self):
        raise_error(NotImplementedError, "TO DO")


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


class QuantumDecodingLayer(QuantumCircuitLayer):
    pass


@dataclass
class ExpectationLayer(QuantumDecodingLayer):

    observable: Union["ndarray", "qibo.models.Hamiltonian"]

    def _feed_input(self, x):
        if isinstance(x, Circuit):
            self.circuit = x

    def forward(self, x):
        if isinstance(x, Circuit):
            return self.observable.expectation_from_samples(
                super().forward(x).frequencies()
            )
        else:
            return self.observable.expectation(x)

    def backward(self):
        raise_error(NotImplementedError, "TO DO")


"""
   .
   .
   .
   .
"""
