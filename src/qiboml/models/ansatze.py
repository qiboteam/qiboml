import random
from dataclasses import dataclass

from qibo import Circuit, gates

from qiboml.models.abstract import QuantumCircuitLayer


@dataclass
class ReuploadingLayer(QuantumCircuitLayer):

    def __post_init__(self):
        super().__post_init__()
        params = self.backend.cast(
            [[random.random() - 0.5 for _ in range(2)] for _ in range(self.nqubits)],
            dtype=self.backend.np.float64,
            requires_grad=True,
        )
        for q, param in zip(self.qubits, params):
            self.circuit.add(gates.RY(q, theta=param[0] * self.backend.np.pi))
            self.circuit.add(gates.RZ(q, theta=param[1] * self.backend.np.pi))
        for i, q in enumerate(self.qubits[:-2]):
            self.circuit.add(gates.CNOT(q0=q, q1=self.qubits[i + 1]))
        self.circuit.add(gates.CNOT(q0=self.qubits[-1], q1=self.qubits[0]))

    def forward(self, x: Circuit) -> Circuit:
        return x + self.circuit
