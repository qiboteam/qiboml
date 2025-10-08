from copy import deepcopy
from typing import Optional

import numpy as np
from qibo import Circuit, gates
from qibo.backends import _check_backend
from qibo.config import raise_error
from qibo.models.encodings import entangling_layer
from scipy.special import binom


def hardware_efficient(
    nqubits: int,
    qubits: Optional[tuple[int]] = None,
    nlayers: int = 1,
    single_block: Optional[Circuit] = None,
    entangling_block: Optional[Circuit] = None,
    entangling_gate: str = "CNOT",
    architecture: str = "diagonal",
    closed_boundary: bool = True,
    seed: Optional[int | np.random.Generator] = None,
    backend=None,
    **kwargs,
) -> Circuit:
    """
    Create a hardware-efficient ansatz with custom single-qubit layers and entangling blocks.

    Args:
        nqubits (int): Number of qubits :math:`n` in the ansatz.
        qubits (tuple[int], optional): Qubit indexes to apply the ansatz to. If ``None``,
            the ansatz is applied to all qubits from :math:`0` to :math:`nqubits-1`.
            Defaults to ``None``.
        nlayers (int, optional): Number of layers (single-qubit + entangling per layer). Defaults to :math:`1`.
        single_block (Circuit, optional): :math:`1`-qubit circuit applied to each qubit.
        If ``None``, defaults to a block with :class:`qibo.gates.RY` and
        :class:`qibo.gates.RZ` gates with Haar-random sampled phases. Defaults to ``None``.
        entangling_block (Circuit, optional): :math:`n`-qubit entangling circuit. Defaults to ``None``.
        entangling_gate (str or :class:`qibo.gates.Gate`, optional): Only used if ``entangling_block``
            is ``None``. Two-qubit gate to be used in the entangling layer if ``entangling_block`` is not
            provided. If ``entangling_gate`` is a parametrized gate, all phases are initialized as
            :math:`0.0`. Defaults to  ``"CNOT"``.
        architecture (str, optional): Only used if ``entangling_block`` is ``None``.
            Architecture of the entangling layer. In alphabetical order, options are:
            ``"diagonal"``, ``"even_layer"``, ``"next_nearest"``, ``"odd_layer"``,
            ``"pyramid"``, ``"shifted"``, ``"v"``, and ``"x"``. The ``"x"`` architecture
            is only defined for an even number of qubits. Defaults to ``"diagonal"``.
        closed_boundary (bool, optional): Only used if ``entangling_block`` is ``None``.
            If ``True`` and ``architecture not in ["pyramid", "v", "x"]``, adds a
            closed-boundary condition to the entangling layer. Defaults to ``True``.
        seed (int or :class:`numpy.random.Generator`, optional): Either a generator of random
            numbers or a fixed seed to initialize a generator. If ``None``, initializes
            a generator with a random seed. Default: ``None``.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be used
            in the execution. If ``None``, it uses the current backend.
            Defaults to ``None``.
        kwargs (dict, optional): Additional arguments used to initialize a Circuit object.
            For details, see the documentation of :class:`qibo.models.circuit.Circuit`.

    Returns:
        :class:`qibo.models.circuit.Circuit`: Constructed hardware-efficient ansatz.
    """
    circ = Circuit(nqubits, **kwargs)

    if qubits is None:
        qubits = list(range(nqubits))
    elif len(qubits) > nqubits:
        raise_error(
            ValueError,
            f"Number of specified qubits ({len(qubits)}) cannot exceed the total number of qubits in the circuit ({nqubits}).",
        )

    if single_block is None:
        from qibo.quantum_info.random_ensembles import uniform_sampling_U3

        backend = _check_backend(backend)
        phases = uniform_sampling_U3(1, seed, backend=backend)[0]
        phases = backend.to_numpy(phases)
        single_block = Circuit(1)
        single_block.add(gates.RY(0, theta=phases[0], trainable=True))
        single_block.add(gates.RZ(0, theta=phases[1], trainable=True))

    for _ in range(nlayers):
        for q in qubits:
            circ.add(single_block.on_qubits(q))

        if len(qubits) != 1:
            if entangling_block is None:
                entangling_block = entangling_layer(
                    nqubits=len(qubits),
                    architecture=architecture,
                    entangling_gate=entangling_gate,
                    closed_boundary=closed_boundary,
                )
            elif entangling_block.nqubits != len(qubits):
                raise_error(
                    ValueError,
                    f"Entangling layer circuit must have {len(qubits)} qubits.",
                )

            circ.add(entangling_block.on_qubits(*qubits))

    return circ


HardwareEfficient = hardware_efficient


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
