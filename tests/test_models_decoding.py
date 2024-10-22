import numpy as np
import pytest
from qibo import gates, hamiltonians
from qibo.quantum_info import random_clifford
from qibo.symbols import Z

import qiboml.models.decoding as dec


def test_probabilities_layer(backend):
    nqubits = 5
    qubits = np.random.choice(range(nqubits), size=(4,), replace=False)
    layer = dec.Probabilities(nqubits, qubits=qubits, backend=backend)
    c = random_clifford(nqubits, backend=backend)
    backend.assert_allclose(
        layer(c).ravel(), backend.execute_circuit(c).probabilities()
    )


def test_state_layer(backend):
    nqubits = 5
    layer = dec.State(nqubits, backend=backend)
    c = random_clifford(nqubits, backend=backend)
    real, im = layer(c)
    backend.assert_allclose(
        (real + 1j * im).ravel(), backend.execute_circuit(c).state().ravel()
    )


@pytest.mark.parametrize("analytic", [True, False])
def test_expectation_layer(backend, analytic):
    backend.set_seed(42)
    rng = np.random.default_rng(42)
    nqubits = 5
    # test observable error
    with pytest.raises(RuntimeError):
        layer = dec.Expectation(nqubits, backend=backend)

    c = random_clifford(nqubits, seed=rng, backend=backend)
    observable = hamiltonians.SymbolicHamiltonian(
        sum([Z(i) for i in range(nqubits)]),
        nqubits=nqubits,
        backend=backend,
    )
    layer = dec.Expectation(
        nqubits,
        observable=observable,
        nshots=int(1e5),
        backend=backend,
        analytic=analytic,
    )
    layer_expv = layer(c)
    expv = (
        observable.expectation(backend.execute_circuit(c).state())
        if analytic
        else observable.expectation_from_samples(
            layer.circuit.measurements[0].result.frequencies()
        )
    )
    backend.assert_allclose(layer_expv, expv)
