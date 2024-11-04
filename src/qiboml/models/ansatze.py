import random

import numpy as np
from qibo import Circuit, gates


def ReuploadingCircuit(nqubits: int, qubits: list[int] = None) -> Circuit:
    if qubits is None:
        qubits = list(range(nqubits))

    circuit = Circuit(nqubits)
    for q in qubits:
        circuit.add(gates.RY(q, theta=random.random() * np.pi, trainable=True))
        circuit.add(gates.RZ(q, theta=random.random() * np.pi, trainable=True))
    for i, q in enumerate(qubits[:-2]):
        circuit.add(gates.CNOT(q0=q, q1=qubits[i + 1]))
    circuit.add(gates.CNOT(q0=qubits[-1], q1=qubits[0]))
    return circuit
