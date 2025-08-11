import numpy as np
import pytest
from qibo import Circuit, gates
from qibo.quantum_info import random_statevector
from scipy.special import binom

from qiboml.models.ansatze import brickwork_givens


@pytest.mark.parametrize("density_matrix", [False, True])
@pytest.mark.parametrize("full_hwp", [False, True])
@pytest.mark.parametrize("weight", [2])
@pytest.mark.parametrize("nqubits", [4])
def test_brickwork_givens(backend, nqubits, weight, full_hwp, density_matrix):
    n_choose_k = int(binom(nqubits, weight))

    np.random.seed(10)
    params = 2 * np.pi * np.random.rand(n_choose_k - 1)
    params = backend.cast(params, dtype="float64")

    qubits = tuple(range(0, nqubits - 1, 2)) + tuple(range(1, nqubits - 1, 2))
    depth = n_choose_k // len(qubits)

    target_circ = Circuit(nqubits, density_matrix=density_matrix)
    if not full_hwp:
        half_filling = nqubits // 2
        _weight = half_filling if weight > half_filling else weight
        target_circ.add(gates.X(2 * qubit) for qubit in range(_weight))
        if weight > half_filling:
            target_circ.add(
                gates.X(2 * qubit + 1) for qubit in range(weight - half_filling)
            )

    _weight = weight if not full_hwp else 0
    for _ in range(depth):
        target_circ.add(
            gates.GIVENS(qubit, qubit + 1, 0.0)
            for qubit in qubits
            if len(target_circ.queue) < n_choose_k + _weight - 1
        )
    target_circ.set_parameters(params)
    target = target_circ.unitary(backend)

    circuit = brickwork_givens(
        nqubits, weight, full_hwp=full_hwp, density_matrix=density_matrix
    )
    circuit.set_parameters(params)
    unitary = circuit.unitary(backend)

    backend.assert_allclose(unitary, target)

    assert target_circ.density_matrix == circuit.density_matrix
