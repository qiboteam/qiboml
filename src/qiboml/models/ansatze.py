import random
from typing import List, Optional

import numpy as np
from qibo import Circuit, gates
from qibo.config import raise_error


def entangling_circuit(nqubits: int, entangling_gate: gates.Gate = gates.CNOT):
    """Construct entangling layer."""
    if nqubits < 2:
        raise_error(ValueError, "This layer cannot be used if nqubits is < 2.")
    circuit = Circuit(nqubits)
    for q in range(nqubits):
        circuit.add(entangling_gate(q0=q % nqubits, q1=(q + 1) % nqubits))
    return circuit


def layered_ansatz(
    nqubits: int,
    nlayers: int = 1,
    qubits: list[int] = None,
    gates_list: Optional[List[gates.Gate]] = [
        gates.RY,
        gates.RZ,
    ],  # TODO: this has to be a circuit
    entanglement: bool = True,
):
    if qubits is None:
        qubits = list(range(nqubits))

    circuit = Circuit(nqubits)
    for _ in range(nlayers):
        for q in qubits:
            for gate in gates_list:
                circuit.add(gate(q, theta=random.random(), trainable=True))
        if entanglement and nqubits > 1:
            circuit += entangling_circuit(nqubits, gates.CNOT)

    return circuit
