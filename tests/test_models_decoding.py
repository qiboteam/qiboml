import numpy as np
import pytest
from qibo import gates, hamiltonians
from qibo.quantum_info import random_clifford
from qibo.symbols import Z

import qiboml.models.encoding_decoding as ed


def test_probabilities_layer(backend):
    nqubits = 5
    qubits = np.random.choice(range(nqubits), size=(4,), replace=False)
    layer = ed.ProbabilitiesLayer(nqubits, qubits=qubits, backend=backend)
    c = random_clifford(nqubits, backend=backend)
    backend.assert_allclose(layer(c).ravel(), c().probabilities(qubits))


def test_state_layer(backend):
    nqubits = 5
    layer = ed.StateLayer(nqubits, backend=backend)
    c = random_clifford(nqubits, backend=backend)
    real, im = layer(c)
    backend.assert_allclose(real + 1j * im, c().state())


@pytest.mark.parametrize("analytic", [True, False])
def test_expectation_layer(backend, analytic):
    backend.set_seed(42)
    nqubits = 5
    # test observable error
    with pytest.raises(RuntimeError):
        layer = ed.ExpectationLayer(nqubits, backend=backend)

    c = random_clifford(nqubits, backend=backend)
    observable = hamiltonians.SymbolicHamiltonian(
        sum([Z(i) for i in range(nqubits)]),
        nqubits=nqubits,
        backend=backend,
    )
    layer = ed.ExpectationLayer(
        nqubits, observable=observable, nshots=int(1e5), backend=backend
    )
    c_copy = c.copy()
    c_copy.add(gates.M(*range(nqubits)))
    expv = (
        observable.expectation(backend.execute_circuit(c_copy, nshots=int(1e5)).state())
        if analytic
        else observable.expectation_from_samples(
            backend.execute_circuit(c_copy, nshots=int(1e5)).frequencies()
        )
    )
    backend.assert_allclose(layer(c), expv, atol=1e-1)