from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import tensorflow as tf
import tensorflow.experimental.numpy as tnp
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

    @property
    def circuit(
        self,
    ):
        return self._circuit


@dataclass
class PhaseEncoding(QuantumEncoding):

    def __post_init__(
        self,
    ):
        super().__post_init__()
        for q in self.qubits:
            self._circuit.add(gates.RZ(q, theta=0.0, trainable=False))

    def _set_phases(self, x: ndarray):
        phase = tf.reshape(x, [-1])
        for i, gate in enumerate(self._circuit.parametrized_gates):
            gate.parameters = phase[i]

    def __call__(self, x: ndarray) -> Circuit:
        self._set_phases(x)
        return self._circuit


@dataclass
class BinaryEncoding(QuantumEncoding):

    def __call__(self, x: ndarray) -> Circuit:
        if x.shape[-1] != len(self.qubits):
            raise_error(
                RuntimeError,
                f"Invalid input dimension {x.shape[-1]}, but the allocated qubits are {self.qubits}.",
            )
        circuit = self.circuit.copy()

        def true_fn():
            circuit.add(gates.X(q))

        def false_fn():
            tf.no_op()

        for i, q in enumerate(self.qubits):
            pred = tf.equal(x[0][i], 1)
            tf.cond(pred, true_fn=true_fn, false_fn=false_fn)

        return circuit
