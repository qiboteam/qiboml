from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from qibo import Circuit, gates
from qibo.config import raise_error

from qiboml import ndarray
from qiboml.models.ansatze import layered_ansatz


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
            self._circuit.add(gates.RY(q, theta=0.0, trainable=False))

    def _set_phases(self, x: ndarray):
        for gate, phase in zip(self._circuit.parametrized_gates, x.ravel()):
            gate.parameters = phase

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
        ones = np.flatnonzero(x.ravel() == 1)
        for bit in ones:
            circuit.add(gates.X(self.qubits[bit]))
        return circuit


@dataclass
class ReuploadingEncoding(QuantumEncoding):
    """
    Implementing reuploading scheme alternating encoding U and training V layers.
    It follows the scheme V - U - V - U - V (in case of 2 layers), namely upload `nlayers` times x and
    and each encoding layer is preceded and followed by a trainable layer V.
    The chosen default V layer is a
    `qiboml.models.ansatze.layered_ansatz(nqubits, 1, qubits, [RY, RZ], True)`.
    """

    # big TODO: make this model more flexible enabling e.g. lambda function of data and params

    # TODO: rm this and raise error when calling the class
    data_shape: tuple = (1,)
    nlayers: int = 1
    trainable_ansatz: Circuit = None
    encoding_gate: gates.Gate = gates.RX

    def __post_init__(
        self,
    ):

        super().__post_init__()

        data_dim = np.prod(self.data_shape, axis=0)

        if int(data_dim) % len(self.qubits) != 0:
            raise_error(
                ValueError,
                f"The data dimension has to be equal to the length of the chosen {self.qubits} subset of the {self.nqubits} system.",
            )

        # TODO: use deepcopy to repeat the call creating single elements
        if self.trainable_ansatz is None:
            self._circuit += layered_ansatz(nqubits=self.nqubits, qubits=self.qubits)
        for _ in range(self.nlayers):
            for q in self.qubits:
                self._circuit.add(self.encoding_gate(q=q, theta=0.0, trainable=False))
            if self.trainable_ansatz is None:
                self._circuit += layered_ansatz(
                    nqubits=self.nqubits, qubits=self.qubits
                )

    def _set_phases(self, x: ndarray):
        encoding_gates = [
            g for g in self._circuit.parametrized_gates if g.trainable == False
        ]
        data_length = len(x.ravel())
        for l in range(self.nlayers):
            for gate, phase in zip(
                encoding_gates[data_length * l : data_length * l + data_length],
                x.ravel(),
            ):
                gate.parameters = phase

    def __call__(self, x: ndarray) -> Circuit:
        self._set_phases(x)
        return self._circuit
