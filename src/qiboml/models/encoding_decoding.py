"""Some standard encoding and decoding layers"""

from dataclasses import dataclass
from typing import Union

import numpy as np
from qibo import Circuit, gates
from qibo.config import raise_error
from qibo.hamiltonians import Hamiltonian

from qiboml import ndarray
from qiboml.models.abstract import QuantumCircuitLayer


@dataclass
class QuantumEncodingLayer(QuantumCircuitLayer):
    pass


@dataclass
class BinaryEncodingLayer(QuantumEncodingLayer):

    def forward(self, x: ndarray) -> Circuit:
        if x.shape[-1] != len(self.qubits):
            raise_error(
                RuntimeError,
                f"Invalid input dimension {x.shape[-1]}, but the allocated qubits are {self.qubits}.",
            )
        circuit = self.circuit.copy()
        ones = np.flatnonzero(x.ravel() == 1)
        for bit in ones:
            circuit.add(gates.X(self.qubits[bit]))
        return circuit


@dataclass
class PhaseEncodingLayer(QuantumEncodingLayer):

    def __post_init__(self):
        super().__post_init__()
        for q in self.qubits:
            self.circuit.add(gates.RZ(q, theta=0.0))

    def forward(self, x: ndarray) -> Circuit:
        self.parameters = x * self.backend.np.pi
        return self.circuit


@dataclass
class QuantumDecodingLayer(QuantumCircuitLayer):

    nshots: int = 1000

    def __post_init__(self):
        super().__post_init__()
        self.circuit.add(gates.M(*self.qubits))

    def forward(self, x: Circuit) -> "CircuitResult":
        return self.backend.execute_circuit(x + self.circuit, nshots=self.nshots)


class ProbabilitiesLayer(QuantumDecodingLayer):

    def __post_init__(self):
        super().__post_init__()

    def forward(self, x: Circuit) -> ndarray:
        return super().forward(x).probabilities(self.qubits).reshape(1, -1)

    @property
    def output_shape(self):
        return (1, 2 ** len(self.qubits))


class SamplesLayer(QuantumDecodingLayer):

    def forward(self, x: Circuit) -> ndarray:
        return self.backend.cast(super().forward(x).samples(), dtype=np.float64)

    @property
    def output_shape(self):
        return (self.nshots, len(self.qubits))


class StateLayer(QuantumDecodingLayer):

    def forward(self, x: Circuit) -> ndarray:
        state = super().forward(x).state()
        return self.backend.np.vstack(
            (self.backend.np.real(state), self.backend.np.imag(state))
        )

    @property
    def output_shape(self):
        return (2, 2**self.nqubits)


@dataclass
class ExpectationLayer(QuantumDecodingLayer):

    observable: Union[ndarray, Hamiltonian] = None
    analytic: bool = False

    def __post_init__(self):
        if self.observable is None:
            raise_error(
                RuntimeError,
                "Please provide an observable for expectation value calculation.",
            )
        super().__post_init__()

    def forward(self, x: Circuit) -> ndarray:
        if self.analytic:
            return self.observable.expectation(
                super().forward(x).state(),
            ).reshape(1, 1)
        else:
            return self.backend.cast(
                self.observable.expectation_from_samples(
                    super().forward(x).frequencies(),
                    qubit_map=self.qubits,
                ).reshape(1, 1),
                dtype=np.float64,
            )

    @property
    def output_shape(self):
        return (1, 1)
