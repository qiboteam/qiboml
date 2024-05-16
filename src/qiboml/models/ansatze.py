from qibo import Circuit, gates

from qiboml.models.abstract import QuantumCircuitLayer


class ReuploadingLayer(QuantumCircuitLayer):

    def _feed_input(self, x):
        if isinstance(x, Circuit):
            self.circuit = self.circuit + x
        else:
            self.initial_state = x

    def __post_init__(self):
        super().__post_init__()
        for q in self.qubits:
            self.circuit.add(gates.RY(q, 0))
            self.circuit.add(gates.RZ(q, 0))
        for i, q in enumerate(self.qubits[:-2]):
            self.circuit.add(gates.CNOT(q0=q, q1=self.qubits[i + 1]))
        self.circuit.add(gates.CNOT(q0=self.qubits[-1], q1=self.qubits[0]))

    def backward(self):
        raise_error(NotImplementedError, "TO DO")
