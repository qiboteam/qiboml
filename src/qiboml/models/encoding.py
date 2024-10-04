from abc import ABC, abstractmethod
from dataclasses import dataclass

from qibo import Circuit, gates

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


class PhaseEncoding(QuantumEncoding):

    def __post_init__(
        self,
    ):
        super().__post_init__()
        for q in self.qubits:
            self._circuit.add(gates.RZ(q, theta=0.0, trainable=False))

    def _set_phases(self, x: ndarray):
        for gate, phase in zip(self._circuit.parametrized_gates, x):
            gate.parameters = phase

    def __call__(self, x: ndarray) -> Circuit:
        self._set_phases(x)
        return self._circuit
