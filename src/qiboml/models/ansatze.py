import random
from copy import deepcopy
from typing import Optional

import numpy as np
from qibo import Circuit, gates
from qibo.models.encodings import entangling_layer
from scipy.special import binom


def HardwareEfficient(
    nqubits: int,
    qubits: Optional[tuple[int]] = None,
    nlayers: int = 1,
) -> Circuit:
    if qubits is None:
        qubits = list(range(nqubits))
    circuit = Circuit(nqubits)

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
    """Create a Hamming-weight-preserving circuit based on brickwork layers of two-qubit Givens rotations.

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

    References:
        1. B. T. Gard, L. Zhu1, G. S. Barron, N. J. Mayhall, S. E. Economou, and Edwin Barnes,
        *Efﬁcient symmetry-preserving state preparation circuits for the variational quantum
        eigensolver algorithm*, `npj Quantum Information (2020) 6:10
        <https://doi.org/10.1038/s41534-019-0240-1>`_.

    """
    n_choose_k = int(binom(nqubits, weight))
    _weight = weight if not full_hwp else 0
    half_filling = nqubits // 2

    circuit = Circuit(nqubits, **kwargs)
    if not full_hwp:
        if weight > half_filling:
            circuit.add(gates.X(2 * qubit) for qubit in range(half_filling))
            circuit.add(
                gates.X(2 * qubit + 1) for qubit in range(weight - half_filling)
            )
        else:
            circuit.add(gates.X(2 * qubit) for qubit in range(weight))

    for _ in range(n_choose_k // (nqubits - 1) - 1):
        circuit += entangling_layer(
            nqubits,
            architecture="shifted",
            entangling_gate=gates.GIVENS,
            closed_boundary=False,
            **kwargs
        )

    ngates = len(circuit.gates_of_type(gates.GIVENS))
    nmissing = (n_choose_k - 1) - ngates

    queue = entangling_layer(
        nqubits,
        architecture="shifted",
        entangling_gate=gates.GIVENS,
        closed_boundary=False,
        **kwargs
    ).queue

    circuit.add(deepcopy(queue[elem % len(queue)]) for elem in range(nmissing))

    return circuit


def FloquetEcho(
    nqubits: int,
    nlayers: int = 2,
    b: float = 0.4 * np.pi,
    theta: float = 0.5 * np.pi,
    decompose_rzz: bool = True,
    target_qubit: int = None,
) -> Circuit:
    """Floquet echo ansatz:
       U = H(target) · (FloquetLayer)^nlayers · RZ(target, theta) · (FloquetLayer^nlayers)†

    Args:
        nqubits: number of qubits.
        nlayers: number of Floquet layers in the 'half-sandwich'.
        b: RX angle used on both qubits of each pair inside each sublayer.
        theta: central RZ angle on the target qubit.
        target_qubit: qubit index for H and central RZ.
        decompose_rzz: if True, decompose RZZ into CNOT–RZ–CNOT.

    Returns:
        Circuit implementing the Floquet echo.
    """

    def _decomposed_rzz(layer_circ: Circuit, q0: int, q1: int, angle: float):
        layer_circ.add(gates.CNOT(q0, q1))
        layer_circ.add(gates.RZ(q=q1, theta=angle))
        layer_circ.add(gates.CNOT(q0, q1))

    def _build_sublayer(nq: int, parity: str) -> Circuit:
        sub = Circuit(nq)
        if nq < 2:
            return sub
        if parity == "even":
            pairs = range(0, nq - 1, 2)
        elif parity == "odd":
            pairs = range(1, nq - 1, 2)
        else:
            raise ValueError("parity must be 'even' or 'odd'.")

        for q1 in pairs:
            q2 = q1 + 1
            sub.add(gates.RZ(q=q1, theta=0.25 * np.pi))
            sub.add(gates.RX(q=q1, theta=b))
            sub.add(gates.RZ(q=q2, theta=0.25 * np.pi))
            sub.add(gates.RX(q=q2, theta=b))
            if decompose_rzz:
                _decomposed_rzz(sub, q1, q2, 0.5 * np.pi)
            else:
                sub.add(gates.RZZ(q0=q1, q1=q2, theta=0.5 * np.pi))
        return sub

    def _build_floquet_layer(nq: int) -> Circuit:
        layer = Circuit(nq)
        layer += _build_sublayer(nq, "even")
        layer += _build_sublayer(nq, "odd")
        return layer

    if target_qubit is None:
        target_qubit = int(nqubits / 2)

    # --- build the circuit ---
    circuit = Circuit(nqubits)
    half_sandwich = Circuit(nqubits)

    # build (FL)^nlayers
    for _ in range(nlayers):
        half_sandwich += _build_floquet_layer(nqubits)

    # H on target
    circuit.add(gates.H(target_qubit))
    # forward half
    circuit += half_sandwich
    # central RZ on target
    circuit.add(gates.RZ(q=target_qubit, theta=theta))
    # inverse half
    circuit += half_sandwich.invert()

    return circuit
