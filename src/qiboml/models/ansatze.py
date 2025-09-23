from copy import deepcopy
import random

import numpy as np
from qibo import Circuit, gates
from qibo.models.encodings import entangling_layer
from scipy.special import binom


def HardwareEfficient(
    nqubits,
    nlayers=1,
    single_block=None,
    entangling_block=None,
    entangling_gate="CNOT",
    architecture="diagonal",
    closed_boundary=False,
    **kwargs,
):
    """
    Create a Hardware-efficient ansatz with custom single-qubit and entangling blocks.

    Args:
        nqubits (int): Number of qubits in the ansatz.
        nlayers (int, optional): Number of layers (single-qubit + entangling per layer). Defaults to 1.
        single_block (Circuit, optional): 1-qubit Circuit applied to each qubit. Defaults to a block with :class:`qibo.gates.RY` and :class:`qibo.gates.RZ` gates.
        entangling_block (Circuit, optional): full n-qubit entangling circuit. Defaults to ``None``.
        entangling_gate (str or :class:`qibo.gates.Gate`, optional): Only used if ``entangling_block`` is None. Two-qubit gate to be used
            in the entangling layer if ``entangling_block`` is not provided. If ``entangling_gate`` is a parametrized gate,
            all phases are initialized as :math:`0.0`. Defaults to  ``"CNOT"``.
        architecture (str, optional): Only used if ``entangling_block`` is None. Architecture of the entangling layer.
            In alphabetical order, options are ``"diagonal"``, ``"even_layer"``,
            ``"next_nearest"``, ``"odd_layer"``, ``"pyramid"``, ``"shifted"``,
            ``"v"``, and ``"x"``. The ``"x"`` architecture is only defined for an even number
            of qubits. Defaults to ``"diagonal"``.
        closed_boundary (bool, optional): Only used if ``entangling_block`` is None. If ``True`` and ``architecture not in
            ["pyramid", "v", "x"]``, adds a closed-boundary condition to the entangling layer.
            Defaults to ``False``.
        kwargs (dict, optional): Additional arguments used to initialize a Circuit object.
            For details, see the documentation of :class:`qibo.models.circuit.Circuit`.

    Returns:
        qibo.models.Circuit: Constructed hardware-efficient ansatz.
    """
    circ = Circuit(nqubits, **kwargs)

    if single_block is None:
        single_block = Circuit(1)
        single_block.add(gates.RY(0, theta=random.random() * np.pi, trainable=True))
        single_block.add(gates.RZ(0, theta=random.random() * np.pi, trainable=True))

    for _ in range(nlayers):
        for q in range(nqubits):
            circ.add(single_block.on_qubits(q))

        if entangling_block is None:
            entangling_block = entangling_layer(
                nqubits=nqubits,
                architecture=architecture,
                entangling_gate=entangling_gate,
                closed_boundary=closed_boundary,
            )
        elif entangling_block.nqubits != nqubits:
            raise ValueError(f"Entangling layer circuit must have {nqubits} qubits.")

        circ += entangling_block

    return circ


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
            **kwargs,
        )

    ngates = len(circuit.gates_of_type(gates.GIVENS))
    nmissing = (n_choose_k - 1) - ngates

    queue = entangling_layer(
        nqubits,
        architecture="shifted",
        entangling_gate=gates.GIVENS,
        closed_boundary=False,
        **kwargs,
    ).queue

    circuit.add(deepcopy(queue[elem % len(queue)]) for elem in range(nmissing))

    return circuit
