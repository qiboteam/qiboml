import math

import pytest
from qibo import Circuit, gates, hamiltonians
from qibo.symbols import X, Z

from qibotn.backends.qmatchatea import QMatchaTeaBackend


def build_observable(nqubits):
    """Helper function to construct a target observable."""
    hamiltonian_form = 0
    for i in range(nqubits):
        hamiltonian_form += 0.5 * X(i % nqubits) * Z((i + 1) % nqubits)

    hamiltonian = hamiltonians.SymbolicHamiltonian(form=hamiltonian_form)
    return hamiltonian, hamiltonian_form


def build_GHZ(nqubits):
    """Helper function to construct a layered quantum circuit."""
    circ = Circuit(nqubits)
    circ.add(gates.H(0))
    [circ.add(gates.CNOT(q, q + 1)) for q in range(nqubits - 1)]
    return circ


def construct_targets(nqubits):
    """Construct strings of 1s and 0s of size `nqubits`."""
    ones = "1" * nqubits
    zeros = "0" * nqubits
    return ones, zeros


@pytest.mark.parametrize("nqubits", [2, 10, 40])
def test_probabilities(backend, nqubits):

    circ = build_GHZ(nqubits=nqubits)

    if isinstance(backend, QMatchaTeaBackend):
        # unbiased prob
        out_u = backend.execute_circuit(
            circuit=circ,
            prob_type="U",
            num_samples=1000,
        ).probabilities()

        math.isclose(out_u[0], 0.5, abs_tol=1e-7)
        math.isclose(out_u[1], 0.5, abs_tol=1e-7)

        out_g = backend.execute_circuit(
            circuit=circ,
            prob_type="G",
            prob_threshold=1.0,
        ).probabilities()

        math.isclose(out_g[0], 0.5, abs_tol=1e-7)
        math.isclose(out_g[1], 0.5, abs_tol=1e-7)

        out_e = backend.execute_circuit(
            circuit=circ,
            prob_type="E",
            prob_threshold=0.2,
        ).probabilities()

        math.isclose(out_e[0], 0.5, abs_tol=1e-7)
        math.isclose(out_e[1], 0.5, abs_tol=1e-7)


@pytest.mark.parametrize("nqubits", [2, 10, 40])
@pytest.mark.parametrize("nshots", [100, 1000])
def test_shots(backend, nqubits, nshots):
    circ = build_GHZ(nqubits=nqubits)
    ones, zeros = construct_targets(nqubits)

    # For p = 0.5, sigma = sqrt(nshots * 0.5 * 0.5) = sqrt(nshots)/2.
    sigma_threshold = 3 * (math.sqrt(nshots) / 2)

    outcome = backend.execute_circuit(circ, nshots=nshots)
    frequencies = outcome.frequencies()

    shots_ones = frequencies.get(ones, 0)
    shots_zeros = frequencies.get(zeros, 0)

    # Check that the counts for both outcomes are within the 3-sigma threshold of nshots/2.
    assert (
        abs(shots_ones - (nshots / 2)) < sigma_threshold
    ), f"Count for {ones} deviates too much: {shots_ones} vs expected {nshots/2}"
    assert (
        abs(shots_zeros - (nshots / 2)) < sigma_threshold
    ), f"Count for {zeros} deviates too much: {shots_zeros} vs expected {nshots/2}"
