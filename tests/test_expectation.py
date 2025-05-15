import math
import random

import pytest
from qibo import Circuit, construct_backend, gates, hamiltonians
from qibo.symbols import X, Z


def build_observable(nqubits):
    """Helper function to construct a target observable."""
    hamiltonian_form = 0
    for i in range(nqubits):
        hamiltonian_form += 0.5 * X(i % nqubits) * Z((i + 1) % nqubits)

    hamiltonian = hamiltonians.SymbolicHamiltonian(form=hamiltonian_form)
    return hamiltonian, hamiltonian_form


def build_circuit(nqubits, nlayers, seed=42):
    """Helper function to construct a layered quantum circuit."""
    random.seed(seed)

    circ = Circuit(nqubits)
    for _ in range(nlayers):
        for q in range(nqubits):
            circ.add(gates.RY(q=q, theta=random.uniform(-math.pi, math.pi)))
            circ.add(gates.RZ(q=q, theta=random.uniform(-math.pi, math.pi)))
        [circ.add(gates.CNOT(q % nqubits, (q + 1) % nqubits) for q in range(nqubits))]
    circ.add(gates.M(*range(nqubits)))
    return circ


@pytest.mark.parametrize("nqubits", [2, 5, 10])
def test_observable_expval(backend, nqubits):
    numpy_backend = construct_backend("numpy")
    ham, ham_form = build_observable(nqubits)
    circ = build_circuit(nqubits=nqubits, nlayers=1)

    exact_expval = numpy_backend.calculate_expectation_state(
        hamiltonian=ham,
        state=circ().state(),
        normalize=False,
    )

    tn_expval = backend.expectation(circuit=circ, observable=ham_form)

    assert math.isclose(exact_expval, tn_expval, abs_tol=1e-7)
