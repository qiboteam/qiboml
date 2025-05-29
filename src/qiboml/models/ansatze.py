import random
from typing import Optional, Union

import numpy as np
from qibo import Circuit, gates

from qiboml.models.utils import _get_wire_names_and_qubits


def HardwareEfficient(
    nqubits: int,
    qubits: Optional[Union[tuple[int], tuple["str"]]] = None,
    nlayers: int = 1,
    density_matrix: Optional[bool] = False,
) -> Circuit:

    qubits, wire_names = _get_wire_names_and_qubits(nqubits, qubits)
    circuit = Circuit(nqubits, density_matrix=density_matrix, wire_names=wire_names)

    for _ in range(nlayers):
        for q in qubits:
            circuit.add(gates.RY(q, theta=random.random() * np.pi, trainable=True))
            circuit.add(gates.RZ(q, theta=random.random() * np.pi, trainable=True))
        if nqubits > 1:
            for i, q in enumerate(qubits[:-2]):
                circuit.add(gates.CNOT(q0=q, q1=qubits[i + 1]))
            circuit.add(gates.CNOT(q0=qubits[-1], q1=qubits[0]))

    return circuit
