import random
from typing import Optional

import numpy as np
from qibo import Circuit, gates


def HardwareEfficient(
    nqubits: int,
    qubits: list[int] = None,
    nlayers: int = 1,
    density_matrix: Optional[bool] = False,
) -> Circuit:
    if qubits is None:
        qubits = list(range(nqubits))

    circuit = Circuit(nqubits, density_matrix=density_matrix)

    for _ in range(nlayers):
        for q in qubits:
            circuit.add(gates.RY(q, theta=random.random() * np.pi, trainable=True))
            circuit.add(gates.RZ(q, theta=random.random() * np.pi, trainable=True))
        if nqubits > 1:
            for i, q in enumerate(qubits[:-2]):
                circuit.add(gates.CNOT(q0=q, q1=qubits[i + 1]))
            circuit.add(gates.CNOT(q0=qubits[-1], q1=qubits[0]))

    return circuit
