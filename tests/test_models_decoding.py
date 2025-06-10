import numpy as np
import pytest
from qibo import Circuit, gates, hamiltonians
from qibo.quantum_info import random_clifford
from qibo.symbols import Z
from qibo.transpiler import NativeGates, Passes, Sabre, Unroller

import qiboml.models.decoding as dec


def test_probabilities_layer(backend):
    nqubits = 5
    qubits = np.random.choice(range(nqubits), size=(4,), replace=False)
    layer = dec.Probabilities(nqubits, qubits=qubits, backend=backend)
    c = random_clifford(nqubits, backend=backend)
    backend.assert_allclose(
        layer(c).ravel(), backend.execute_circuit(c).probabilities(qubits)
    )


def test_state_layer(backend):
    nqubits = 5
    layer = dec.State(nqubits, backend=backend)
    c = random_clifford(nqubits, backend=backend)
    real, im = layer(c)
    backend.assert_allclose(
        (real + 1j * im).ravel(), backend.execute_circuit(c).state().ravel()
    )


@pytest.mark.parametrize("nshots", [None, 10000])
@pytest.mark.parametrize(
    "observable",
    [
        None,
        lambda n, b: hamiltonians.SymbolicHamiltonian(
            sum([Z(i) for i in range(n)]), nqubits=n, backend=b
        ),
    ],
)
def test_expectation_layer(backend, nshots, observable):
    backend.set_seed(42)
    rng = np.random.default_rng(42)
    nqubits = 5

    c = random_clifford(nqubits, seed=rng, backend=backend)
    if observable is not None:
        observable = observable(nqubits, backend)
    layer = dec.Expectation(
        nqubits,
        observable=observable,
        nshots=nshots,
        backend=backend,
    )
    layer_expv = layer(c)
    if observable is None:
        observable = hamiltonians.Z(nqubits, dense=False, backend=backend)
    expv = (
        observable.expectation(backend.execute_circuit(c).state())
        if nshots is None
        else observable.expectation_from_samples(
            layer.circuit.measurements[0].result.frequencies()
        )
    )
    backend.assert_allclose(layer_expv, expv)


def test_decoding_with_transpiler(backend):
    rng = np.random.default_rng(42)
    backend.set_seed(42)
    c = random_clifford(3, seed=rng, backend=backend)
    transpiler = Passes(
        connectivity=[[0, 1], [0, 2]], passes=[Unroller(NativeGates.default(), Sabre())]
    )
    layer = dec.Probabilities(3, transpiler=transpiler, backend=backend)
    backend.assert_allclose(backend.execute_circuit(c).probabilities(), layer(c))


def test_decoding_wire_names(backend):
    c = Circuit(3)
    wire_names = ["a", "b", "c"]
    layer = dec.Probabilities(3, wire_names=wire_names, backend=backend)
    layer(c)
    assert c.wire_names == wire_names
    assert list(layer.wire_names) == wire_names
    assert layer.circuit.wire_names == wire_names
