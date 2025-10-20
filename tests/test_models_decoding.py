import numpy as np
import pytest
import torch
from qibo import Circuit, gates, hamiltonians
from qibo.backends import NumpyBackend
from qibo.models.encodings import comp_basis_encoder
from qibo.quantum_info import random_clifford
from qibo.symbols import X, Z
from qibo.transpiler import NativeGates, Passes, Sabre, Unroller

import qiboml.models.decoding as dec
from qiboml.interfaces.pytorch import QuantumModel


def test_probabilities_layer(backend):
    nqubits = 5
    qubits = np.random.choice(range(nqubits), size=(4,), replace=False)
    layer = dec.Probabilities(nqubits, qubits=qubits, backend=backend)
    circuit = random_clifford(nqubits, backend=NumpyBackend())
    backend.assert_allclose(
        layer(circuit).ravel(), backend.execute_circuit(circuit).probabilities(qubits)
    )


@pytest.mark.parametrize("density_matrix", [True, False])
def test_state_layer(backend, density_matrix):
    nqubits = 5
    layer = dec.State(nqubits, density_matrix=density_matrix, backend=backend)
    circuit = random_clifford(
        nqubits, density_matrix=density_matrix, backend=NumpyBackend()
    )
    real, im = layer(circuit)
    backend.assert_allclose(
        (real + 1j * im).ravel(), backend.execute_circuit(circuit).state().ravel()
    )


@pytest.mark.parametrize("nshots", [None, 10000])
@pytest.mark.parametrize(
    "observable",
    [None, hamiltonians.TFIM],
)
def test_expectation_layer(backend, nshots, observable):
    backend.set_seed(42)
    nqubits = 5

    circuit = comp_basis_encoder("1" * 5)

    if observable is not None:
        observable = observable(nqubits, 0.1, False, backend)
    layer = dec.Expectation(
        nqubits,
        observable=observable,
        nshots=nshots,
        backend=backend,
    )
    layer_expv = layer(circuit)
    if observable is None:
        observable = hamiltonians.Z(nqubits, dense=False, backend=backend)
    expv = (
        observable.expectation(backend.execute_circuit(circuit).state())
        if nshots is None
        else observable.expectation_from_circuit(circuit, nshots=nshots)
    )
    atol = 1e-8 if nshots is None else 1e-2

    layer_expv = layer_expv[0, 0]

    backend.assert_allclose(layer_expv, expv, atol=atol)


def test_decoding_with_transpiler(backend):
    rng = np.random.default_rng(42)
    backend.set_seed(42)
    c = random_clifford(3, seed=rng, backend=backend)
    transpiler = Passes(
        connectivity=[[0, 1], [0, 2]], passes=[Unroller(NativeGates.default(), Sabre())]
    )
    layer = dec.Probabilities(3, transpiler=transpiler, backend=backend)
    backend.assert_allclose(
        backend.execute_circuit(c).probabilities(), layer(c).ravel()
    )


def test_decoding_wire_names(backend):
    c = Circuit(3)
    wire_names = ["a", "b", "c"]
    layer = dec.Probabilities(3, wire_names=wire_names, backend=backend)
    layer(c)
    assert c.wire_names == wire_names
    assert list(layer.wire_names) == wire_names
    assert layer.circuit.wire_names == wire_names


def test_vqls_solver_basic(backend):
    """Test the VariationalQuantumLinearSolver on a 1-qubit system."""
    nqubits = 1

    A = backend.cast([[1.0, 0.2], [0.2, 1.0]], dtype=backend.complex128)
    target_state = backend.cast([1.0, 0.0], dtype=backend.complex128)
    circuit = Circuit(nqubits)

    solver = dec.VariationalQuantumLinearSolver(
        nqubits=nqubits,
        target_state=target_state,
        A=A,
        backend=backend,
    )

    cost = solver(circuit)
    assert 0.0 <= cost <= 1.0
    assert solver.output_shape == (1, 1)
    assert solver.analytic
    if backend.platform == "pytorch":

        weights = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float64))]
        )

        def variational_block(weights):
            circuit = Circuit(nqubits)
            circuit.add(gates.H(0))
            circuit.add(gates.RY(0, weights[0]))
            return circuit

        vqc = variational_block(weights)

        model = QuantumModel(
            decoding=solver,
            circuit_structure=vqc,
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
        initial_cost = float(model())

        for _ in range(100):
            optimizer.zero_grad()
            loss = model()
            loss.backward()
            optimizer.step()

        final_cost = float(model())
        assert final_cost < initial_cost
