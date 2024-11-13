from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from qibo import Circuit, gates
from qibo.config import raise_error

from qiboml import ndarray


@dataclass
class QuantumEncoding(ABC):

    nqubits: int
    qubits: list[int] = None
    _circuit: Circuit = None

    def __post_init__(
        self,
    ):
        if self.qubits is None:
            self.qubits = list(range(self.nqubits))
        self._circuit = Circuit(self.nqubits)

    @abstractmethod
    def __call__(self, x: ndarray) -> Circuit:
        pass

    @abstractmethod
    def gates_encoding_feature(self, feature_index: int):
        pass

    @property
    def circuit(
        self,
    ):
        return self._circuit

    @property
    def hardware_differentiable(self):
        return False


@dataclass
class PhaseEncoding(QuantumEncoding):

    def __post_init__(
        self,
    ):
        super().__post_init__()
        for q in self.qubits:
            self._circuit.add(gates.RY(q, theta=0.0, trainable=False))

    def _set_phases(self, x: ndarray):
        for gate, phase in zip(self._circuit.parametrized_gates, x.ravel()):
            gate.parameters = phase

    def __call__(self, x: ndarray) -> Circuit:
        self._set_phases(x)
        return self._circuit

    def gates_encoding_feature(self, feature_index: int):
        return [
            [g for g in self._circuit.parametrized_gates if g.trainable == False][
                feature_index
            ]
        ]

    @property
    def hardware_differentiable(self):
        return True


@dataclass
class BinaryEncoding(QuantumEncoding):

    def __call__(self, x: ndarray) -> Circuit:
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
