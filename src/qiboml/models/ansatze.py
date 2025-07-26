import random
from typing import Optional

import numpy as np
from scipy.special import binom

from qibo import Circuit, gates
from qibo.models.encodings import entangling_layer, phase_encoder


def HardwareEfficient(
    nqubits: int,
    qubits: Optional[tuple[int]] = None,
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


def brickwork_givens(nqubits: int, weight: int, full_hwp: bool = False, **kwargs):
    """Create a Hamming-weight-preserving based on brickwork layers of two-qubit Givens rotations.

    Args:
        nqubits (int): Total number of qubits.
        weight (int): Hamming weight to be encoded.
        full_hwp (bool, optional): If ``False``, returns circuit with the necessary
            :class:`qibo.gates.X` gates included to generate the initial Hamming weight
            to be preserved. If ``True``, circuit does not include the :class:`qibo.gates.X`
            gates. Defaults to ``False``.
        kwargs (dict, optional): Additional arguments used to initialize a Circuit object.
            For details, see the documentation of :class:`qibo.models.circuit.Circuit`.

    Returns:
        :class:`qibo.models.circuit.Circuit`: Hamming-weight-preserving brickwork circuit.
    """
    n_choose_k = int(binom(nqubits, weight))

    circuit = Circuit(nqubits, **kwargs)

    if not full_hwp:
        half_filling = nqubits // 2
        if weight > half_filling:
            circuit.add(gates.X(2 * qubit) for qubit in range(half_filling))
            circuit.add(
                gates.X(2 * qubit + 1) for qubit in range(weight - half_filling)
            )
        else:
            circuit.add(gates.X(2 * qubit) for qubit in range(weight))

    for _ in range(n_choose_k // (nqubits - 1)):
        circuit += entangling_layer(
            nqubits,
            architecture="shifted",
            entangling_gate=gates.GIVENS,
            closed_boundary=False,
            **kwargs
        )

    return circuit
