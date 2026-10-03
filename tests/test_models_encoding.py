import math

import numpy as np
import pytest
import scipy
from qibo import Circuit, gates, hamiltonians
from qibo.backends import NumpyBackend
from qibo.models.encodings import _generate_rbs_angles
from qibo.quantum_info import random_statevector
from scipy.sparse import csr_matrix
from scipy.special import comb

from qiboml.differentiations import PSR, Adjoint
from qiboml.models.decoding import Expectation, State
from qiboml.models.optimizers import ExactGeodesicTransportCG, QuantumNaturalGradient

DIFFERENTIATIONS = [PSR, Adjoint]
try:
    from qiboml.differentiations import Jax

    DIFFERENTIATIONS.append(Jax)
except ImportError:  # pragma: no cover
    Jax = None

GATES = {
    "RX": lambda: gates.RX(0, 0.4),
    "U1": lambda: gates.U1(0, 0.4),
    "U2": lambda: gates.U2(0, 0.4, 0.9),
    "U3": lambda: gates.U3(1, 0.4, 0.9, 1.3),
    "RBS": lambda: gates.RBS(0, 1, 0.4),
    "GIVENS": lambda: gates.GIVENS(0, 1, 0.4),
    "RXX": lambda: gates.RXX(0, 1, 0.4),
    "RYY": lambda: gates.RYY(0, 1, 0.4),
    "RZZ": lambda: gates.RZZ(0, 1, 0.4),
    "RZX": lambda: gates.RZX(0, 1, 0.4),
    "RXXYY": lambda: gates.RXXYY(0, 1, 0.4),
    "CRX": lambda: gates.CRX(0, 1, 0.4),
    "CRY": lambda: gates.CRY(0, 1, 0.4),
    "CRZ": lambda: gates.CRZ(0, 1, 0.4),
}


def test_egt_cg_errors(backend):

    nqubits = 4
    hamiltonian, _ = _get_xxz_hamiltonian(nqubits, "sparse", backend)

    with pytest.raises(TypeError):
        # loss_fn must be callable or str
        loss_fn = 13
        _ = ExactGeodesicTransportCG(
            nqubits=nqubits,
            weight=int(nqubits / 2),
            loss_fn=loss_fn,
            loss_kwargs={"hamiltonian": hamiltonian},
            initial_parameters=None,
            c1=0.485,
            c2=0.999,
            backtrack_rate=0.5,
            backtrack_multiplier=1.5,
            callback=None,
            seed=13,
            backend=backend,
        )
    with pytest.raises(ValueError):
        # if loss_fn is str, it must be "exp_val"
        loss_fn = "expval"
        _ = ExactGeodesicTransportCG(
            nqubits=nqubits,
            weight=int(nqubits / 2),
            loss_fn=loss_fn,
            loss_kwargs={"hamiltonian": hamiltonian},
            initial_parameters=None,
            c1=0.485,
            c2=0.999,
            backtrack_rate=0.5,
            backtrack_multiplier=1.5,
            callback=None,
            seed=13,
            backend=backend,
        )
    with pytest.raises(TypeError):
        # hamiltonian must be ArrayLike or sparse of the given backend
        loss_fn = "exp_val"
        hamiltonian = [
            (1.0, "X" * nqubits),
            (1.0, "Y" * nqubits),
            (1.0, "Z" * nqubits),
        ]
        _ = ExactGeodesicTransportCG(
            nqubits=nqubits,
            weight=int(nqubits / 2),
            loss_fn=loss_fn,
            loss_kwargs={"hamiltonian": hamiltonian},
            initial_parameters=None,
            c1=0.485,
            c2=0.999,
            backtrack_rate=0.5,
            backtrack_multiplier=1.5,
            callback=None,
            seed=13,
            backend=backend,
        )
    with pytest.raises(ValueError):
        # loss_kwargs must have `"hamiltonian": hamiltonian` item if `loss_fn = "exp_val"`
        loss_fn = "exp_val"
        _ = ExactGeodesicTransportCG(
            nqubits=nqubits,
            weight=int(nqubits / 2),
            loss_fn=loss_fn,
            loss_kwargs={"hamiltonian_wrong": hamiltonian},
            initial_parameters=None,
            c1=0.485,
            c2=0.999,
            backtrack_rate=0.5,
            backtrack_multiplier=1.5,
            callback=None,
            seed=13,
            backend=backend,
        )
    with pytest.raises(TypeError):
        # if loss_fn is a callable, we must use a backend that has autodiff
        loss_fn = _loss_func_expval
        backend = NumpyBackend()
        hamiltonian, _ = _get_xxz_hamiltonian(nqubits, "sparse", backend)
        _ = ExactGeodesicTransportCG(
            nqubits=nqubits,
            weight=int(nqubits / 2),
            loss_fn=loss_fn,
            loss_kwargs={"hamiltonian": hamiltonian},
            initial_parameters=None,
            c1=0.485,
            c2=0.999,
            backtrack_rate=0.5,
            backtrack_multiplier=1.5,
            callback=None,
            seed=13,
            backend=backend,
        )


@pytest.mark.parametrize("nqubits", [4, 6])
@pytest.mark.parametrize("hamiltonian_type", ["sparse", "dense"])
@pytest.mark.parametrize("type_loss_grad", ["exp_val", "callable"])
@pytest.mark.parametrize("initial_parameters", ["explicit_HR", None])
def test_egt_cg(
    backend,
    nqubits,
    hamiltonian_type,
    type_loss_grad,
    initial_parameters,
):
    if backend.platform in ["jax", "tensorflow"] and nqubits != 4:
        pytest.skip(
            "Tests too slow with Jax and TF, will test only small size."
            + " Will test all sizes with torch."
        )

    hamiltonian, true_gs_energy = _get_xxz_hamiltonian(
        nqubits, hamiltonian_type, backend
    )
    chem_acc = 0.03 / (27.2114 * backend.abs(true_gs_energy))

    loss_fn = _loss_func_expval if type_loss_grad == "callable" else type_loss_grad

    if initial_parameters == "explicit_HR":
        initial_parameters = _generate_rbs_angles(
            random_statevector(
                int(comb(nqubits, nqubits // 2)),
                dtype=backend.float64,
                seed=13,
                backend=backend,
            ),
            "diagonal",
            backend=backend,
        )

    def make_callback(print_every):
        def callback(
            iter_num,
            loss,
            **kwargs,
        ):
            if iter_num % print_every == 0:
                print(f"Iter {iter_num}: loss = {loss:.6f}")

        return callback

    optimizer = ExactGeodesicTransportCG(
        nqubits=nqubits,
        weight=int(nqubits / 2),
        loss_fn=loss_fn,
        loss_kwargs={"hamiltonian": hamiltonian},
        initial_parameters=initial_parameters,
        c1=0.485,
        c2=0.999,
        backtrack_rate=0.5,
        backtrack_multiplier=1.5,
        callback=make_callback(1000),
        seed=13,
        backend=backend,
    )
    _, losses, _ = optimizer(steps=20)
    rel_errors = backend.abs(1 - losses / true_gs_energy)

    assert rel_errors[-1] < chem_acc


def _loss_func_expval(circuit, backend, *, hamiltonian, weight=None):
    kwargs = {"weight": weight} if backend.name == "hamming_weight" else {}
    psi = backend.execute_circuit(circuit, **kwargs).state()
    platform = backend.platform
    if platform == "tensorflow":
        if "cpu" in backend.device.lower():
            psi_col = backend.reshape(psi, (-1, 1))
            h_psi = backend.engine.sparse.sparse_dense_matmul(hamiltonian, psi_col)
            h_psi = backend.reshape(h_psi, (-1,))
        else:
            psi_col = backend.reshape(psi, (-1, 1))
            h_psi = backend.matmul(backend.engine.sparse.to_dense(hamiltonian), psi_col)
            h_psi = backend.reshape(h_psi, (-1,))

    elif platform == "pytorch":
        h_psi = backend.engine.sparse.mm(hamiltonian, psi.unsqueeze(1)).squeeze(1)
    else:
        h_psi = hamiltonian @ psi

    return backend.real(backend.sum(backend.conj(psi) * h_psi))


@pytest.mark.parametrize("nqubits", [4, 6])
@pytest.mark.parametrize("hamiltonian_type", ["sparse", "dense"])
@pytest.mark.parametrize("type_loss_grad", ["exp_val"])
@pytest.mark.parametrize("initial_parameters", ["explicit_HR", None])
def test_egt_cg_numpy(
    nqubits,
    hamiltonian_type,
    type_loss_grad,
    initial_parameters,
):

    backend = NumpyBackend()

    hamiltonian, true_gs_energy = _get_xxz_hamiltonian(
        nqubits, hamiltonian_type, backend
    )
    chem_acc = 0.03 / (27.2114 * backend.abs(true_gs_energy))

    loss_fn = type_loss_grad

    if initial_parameters == "explicit_HR":
        initial_parameters = _generate_rbs_angles(
            random_statevector(
                int(comb(nqubits, nqubits // 2)),
                dtype=backend.float64,
                seed=13,
                backend=backend,
            ),
            "diagonal",
            backend=backend,
        )

    def make_callback(print_every):
        def callback(
            iter_num,
            loss,
            **kwargs,
        ):
            if iter_num % print_every == 0:
                print(f"Iter {iter_num}: loss = {loss:.6f}")

        return callback

    optimizer = ExactGeodesicTransportCG(
        nqubits=nqubits,
        weight=int(nqubits / 2),
        loss_fn=loss_fn,
        loss_kwargs={"hamiltonian": hamiltonian},
        initial_parameters=initial_parameters,
        c1=0.485,
        c2=0.999,
        backtrack_rate=0.5,
        backtrack_multiplier=1.5,
        callback=make_callback(1000),
        seed=13,
        backend=backend,
    )
    _, losses, _ = optimizer(steps=20)
    rel_errors = backend.abs(1 - losses / true_gs_energy)

    assert rel_errors[-1] < chem_acc


def test_qng_errors(backend):
    hamiltonian = hamiltonians.Z(2, backend=backend)
    decoding = Expectation(nqubits=2, observable=hamiltonian, backend=backend)
    circuit = Circuit(2)
    circuit.add(gates.RX(0, 0.1))

    with pytest.raises(TypeError):
        # decoding must be an Expectation
        _ = QuantumNaturalGradient(circuit, decoding=State(2, backend=backend))
    with pytest.raises(TypeError):
        # differentiation must be a Differentiation class
        _ = QuantumNaturalGradient(circuit, decoding, differentiation="psr")
    with pytest.raises(ValueError):
        # the circuit must have trainable parameters
        _ = QuantumNaturalGradient(Circuit(2), decoding)
    for metric_nshots in (0, -10, 1.5):
        with pytest.raises(ValueError):
            # the number of shots must be a positive integer
            _ = QuantumNaturalGradient(circuit, decoding, metric_nshots=metric_nshots)

    circuit = Circuit(2)
    circuit.add(gates.fSim(0, 1, 0.1, 0.2))
    with pytest.raises(NotImplementedError):
        # trainable gates must be decomposable into Pauli rotations
        _ = QuantumNaturalGradient(circuit, decoding)


@pytest.mark.parametrize("differentiation", DIFFERENTIATIONS)
def test_qng_gradient_and_metric(backend, differentiation):
    if differentiation is Jax and backend.platform != "jax":
        pytest.skip("Jax differentiation is tested only with the jax backend.")
    if differentiation is Adjoint and backend.platform == "tensorflow":
        pytest.skip("Adjoint differentiation uses ``vdot``, missing in tensorflow.")

    theta_1, theta_2 = 0.7, 0.3
    circuit = Circuit(1)
    circuit.add(gates.RX(0, theta_1))
    circuit.add(gates.RY(0, theta_2))

    decoding = Expectation(
        nqubits=1, observable=hamiltonians.Z(1, backend=backend), backend=backend
    )
    optimizer = QuantumNaturalGradient(circuit, decoding, differentiation)
    gradient, metric = optimizer._gradient_and_metric()

    # qibo's Z hamiltonian is minus the sum of Pauli Z operators, so the loss is
    # -cos(theta_1) * cos(theta_2), and the metric is diag(1, cos(theta_1)^2) / 4
    target_gradient = [
        math.sin(theta_1) * math.cos(theta_2),
        math.cos(theta_1) * math.sin(theta_2),
    ]
    target_metric = [[0.25, 0.0], [0.0, math.cos(theta_1) ** 2 / 4]]

    backend.assert_allclose(
        gradient, backend.cast(target_gradient, dtype=backend.float64), atol=1e-8
    )
    backend.assert_allclose(
        metric, backend.cast(target_metric, dtype=backend.float64), atol=1e-8
    )


@pytest.mark.parametrize("differentiation", DIFFERENTIATIONS)
@pytest.mark.parametrize("gate", sorted(GATES))
def test_qng_gates(gate, differentiation):
    backend = NumpyBackend()
    hamiltonian = hamiltonians.TFIM(2, h=0.7, backend=backend)
    decoding = Expectation(nqubits=2, observable=hamiltonian, backend=backend)

    circuit = Circuit(2)
    circuit.add(gates.RY(0, 0.8))
    circuit.add(gates.RX(1, 0.5))
    circuit.add(gates.CNOT(0, 1))
    circuit.add(gates.RY(1, 1.1))
    circuit.add(GATES[gate]())
    circuit.add(gates.RX(0, 0.3))
    circuit.add(gates.RZ(1, 0.6))

    optimizer = QuantumNaturalGradient(circuit, decoding, differentiation)
    gradient, metric = optimizer._gradient_and_metric()

    # reference gradient and metric tensor from finite differences of the original circuit
    parameters = np.array([p for g in circuit.trainable_gates for p in g.parameters])
    step = 1e-5

    def evaluate(shift):
        circuit.set_parameters(list(parameters + shift))
        state = np.array(backend.execute_circuit(circuit).state())
        return state, float(np.real(decoding(circuit)[0, 0]))

    state, _ = evaluate(0.0)
    derivatives, target_gradient = [], []
    for k in range(len(parameters)):
        shift = np.zeros(len(parameters))
        shift[k] = step
        forward, loss_forward = evaluate(shift)
        backward, loss_backward = evaluate(-shift)
        derivatives.append((forward - backward) / (2 * step))
        target_gradient.append((loss_forward - loss_backward) / (2 * step))
    derivatives = np.array(derivatives)
    projections = derivatives.conj() @ state
    target_metric = np.real(
        derivatives.conj() @ derivatives.T - np.outer(projections, projections.conj())
    )
    circuit.set_parameters(list(parameters))

    # the decomposed circuit prepares the same state up to a global phase
    decomposed_state = np.array(backend.execute_circuit(optimizer._decomposed).state())
    backend.assert_allclose(abs(np.vdot(decomposed_state, state)), 1.0, atol=1e-10)
    backend.assert_allclose(gradient, target_gradient, atol=1e-7)
    backend.assert_allclose(metric, target_metric, atol=1e-7)


@pytest.mark.parametrize("differentiation", DIFFERENTIATIONS)
def test_qng_sampled_metric(backend, differentiation):
    if differentiation is Jax and backend.platform != "jax":
        pytest.skip("Jax differentiation is tested only with the jax backend.")
    if differentiation is Adjoint and backend.platform == "tensorflow":
        pytest.skip("Adjoint differentiation uses ``vdot``, missing in tensorflow.")

    nqubits = 2
    hamiltonian = hamiltonians.TFIM(nqubits, h=0.7, backend=backend)
    decoding = Expectation(nqubits=nqubits, observable=hamiltonian, backend=backend)

    circuit = Circuit(nqubits)
    circuit.add(gates.RY(0, 0.8))
    circuit.add(gates.U3(1, 0.5, 0.9, 1.3))
    circuit.add(gates.CNOT(0, 1))
    circuit.add(gates.RBS(0, 1, 0.4))
    circuit.add(gates.U2(0, 0.3, 0.6))

    exact = QuantumNaturalGradient(circuit.copy(deep=True), decoding, differentiation)
    gradient, target_metric = exact._gradient_and_metric()

    nshots = 20000
    backend.set_seed(42)
    sampled = QuantumNaturalGradient(
        circuit.copy(deep=True), decoding, differentiation, metric_nshots=nshots
    )
    sampled_gradient, metric = sampled._gradient_and_metric()

    # the gradient does not depend on the metric, and each entry of the metric tensor
    # has a statistical error of order 1 / sqrt(nshots)
    backend.assert_allclose(sampled_gradient, gradient, atol=1e-8)
    backend.assert_allclose(metric, target_metric, atol=2e-2)
    backend.assert_allclose(metric, backend.transpose(metric, (1, 0)), atol=1e-12)


def test_qng_sampled_metric_error_decreases(backend):
    decoding = Expectation(
        nqubits=2,
        observable=hamiltonians.TFIM(2, h=0.7, backend=backend),
        backend=backend,
    )
    circuit = Circuit(2)
    circuit.add(gates.RY(0, 0.8))
    circuit.add(gates.RX(1, 0.5))
    circuit.add(gates.CNOT(0, 1))
    circuit.add(gates.RY(1, 1.1))

    _, target_metric = QuantumNaturalGradient(
        circuit.copy(deep=True), decoding
    )._gradient_and_metric()

    errors = []
    for nshots in (100, 10000):
        backend.set_seed(7)
        optimizer = QuantumNaturalGradient(
            circuit.copy(deep=True), decoding, metric_nshots=nshots
        )
        _, metric = optimizer._gradient_and_metric()
        errors.append(float(backend.max(backend.abs(metric - target_metric))))

    assert errors[1] < errors[0]


def test_qng_sampled_gradient_and_metric():
    # the gradient is sampled by setting the number of shots of the decoding, and
    # this requires a symbolic hamiltonian
    backend = NumpyBackend()
    nqubits = 2
    nshots = 5000
    hamiltonian = hamiltonians.TFIM(nqubits, h=1.0, dense=False, backend=backend)
    true_gs_energy = backend.real(
        backend.eigenvalues(hamiltonians.TFIM(nqubits, h=1.0, backend=backend).matrix)[
            0
        ]
    )
    decoding = Expectation(
        nqubits=nqubits, observable=hamiltonian, backend=backend, nshots=nshots
    )

    circuit = Circuit(nqubits)
    circuit.add(gates.RY(0, 0.5))
    circuit.add(gates.RY(1, 1.1))
    circuit.add(gates.CNOT(0, 1))
    circuit.add(gates.RY(0, 0.9))
    circuit.add(gates.RY(1, 0.3))
    circuit.add(gates.RX(0, 0.2))
    circuit.add(gates.RX(1, 0.7))

    backend.set_seed(5)
    optimizer = QuantumNaturalGradient(
        circuit,
        decoding,
        metric_nshots=nshots,
        learning_rate=0.1,
        regularization=1e-1,
    )
    _, _, parameters = optimizer(steps=40)

    # energy of the final state without sampling noise
    circuit.set_parameters(parameters)
    state = backend.execute_circuit(circuit).state()
    energy = backend.real(
        hamiltonians.TFIM(nqubits, h=1.0, backend=backend).expectation_from_state(state)
    )
    backend.assert_allclose(energy, true_gs_energy, atol=5e-2)


@pytest.mark.parametrize("nqubits", [2, 4])
def test_qng(backend, nqubits):
    hamiltonian = hamiltonians.TFIM(nqubits, h=1.0, backend=backend)
    true_gs_energy = backend.real(
        backend.cast(backend.eigenvalues(hamiltonian.matrix)[0])
    )
    decoding = Expectation(nqubits=nqubits, observable=hamiltonian, backend=backend)

    circuit = Circuit(nqubits)
    for layer in range(3):
        for qubit in range(nqubits):
            circuit.add(gates.RY(qubit, 0.5 * (layer + 1) + 0.3 * qubit))
        for qubit in range(nqubits):
            circuit.add(gates.CNOT(qubit, (qubit + 1) % nqubits))
    for qubit in range(nqubits):
        circuit.add(gates.RY(qubit, 0.2 * (qubit + 1)))

    iterations = []

    def callback(iter_num, loss, parameters):
        iterations.append(iter_num)

    optimizer = QuantumNaturalGradient(
        circuit, decoding, learning_rate=0.1, callback=callback
    )
    final_loss, losses, parameters = optimizer(steps=100)

    assert len(losses) >= len(iterations)
    assert len(parameters) == len(circuit.trainable_gates)
    backend.assert_allclose(final_loss, losses[-1])
    backend.assert_allclose(final_loss, true_gs_energy, atol=1e-2)


@pytest.mark.parametrize("differentiation", DIFFERENTIATIONS)
def test_qng_vqe_multiparameter_gates(backend, differentiation):
    if differentiation is Jax and backend.platform != "jax":
        pytest.skip("Jax differentiation is tested only with the jax backend.")
    if differentiation is Adjoint and backend.platform == "tensorflow":
        pytest.skip("Adjoint differentiation uses ``vdot``, missing in tensorflow.")

    nqubits = 2
    hamiltonian = hamiltonians.TFIM(nqubits, h=1.0, backend=backend)
    true_gs_energy = backend.real(
        backend.cast(backend.eigenvalues(hamiltonian.matrix)[0])
    )
    decoding = Expectation(nqubits=nqubits, observable=hamiltonian, backend=backend)

    circuit = Circuit(nqubits)
    circuit.add(gates.U3(0, 0.5, 0.3, 0.2))
    circuit.add(gates.U2(1, 1.1, 0.7))
    circuit.add(gates.RBS(0, 1, 0.4))
    circuit.add(gates.U3(0, 0.9, 0.1, 0.6))
    circuit.add(gates.U3(1, 0.3, 0.8, 1.0))

    optimizer = QuantumNaturalGradient(
        circuit, decoding, differentiation, learning_rate=0.1
    )
    final_loss, _, parameters = optimizer(steps=100)

    assert len(parameters) == 3 + 2 + 1 + 3 + 3
    # the circuit passed by the user is updated in place
    backend.assert_allclose(
        backend.cast(
            [p for gate in circuit.trainable_gates for p in gate.parameters],
            dtype=backend.float64,
        ),
        parameters,
    )
    backend.assert_allclose(final_loss, true_gs_energy, atol=1e-2)


def test_qng_sampled_metric_vqe(backend):
    nqubits = 2
    hamiltonian = hamiltonians.TFIM(nqubits, h=1.0, backend=backend)
    true_gs_energy = backend.real(
        backend.cast(backend.eigenvalues(hamiltonian.matrix)[0])
    )
    decoding = Expectation(nqubits=nqubits, observable=hamiltonian, backend=backend)

    circuit = Circuit(nqubits)
    circuit.add(gates.U2(0, 0.5, 0.3))
    circuit.add(gates.RY(1, 1.1))
    circuit.add(gates.RBS(0, 1, 0.4))
    circuit.add(gates.RY(0, 0.9))
    circuit.add(gates.RY(1, 0.3))

    backend.set_seed(11)
    optimizer = QuantumNaturalGradient(
        circuit, decoding, metric_nshots=2000, learning_rate=0.1, regularization=1e-1
    )
    final_loss, _, _ = optimizer(steps=40)

    backend.assert_allclose(final_loss, true_gs_energy, atol=5e-2)


def test_qng_tolerance(backend):
    # a loss that is stationary at the initial parameters stops the optimization at once
    circuit = Circuit(1)
    circuit.add(gates.RY(0, 0.0))
    decoding = Expectation(
        nqubits=1, observable=hamiltonians.Z(1, backend=backend), backend=backend
    )
    optimizer = QuantumNaturalGradient(circuit, decoding)
    _, losses, parameters = optimizer(steps=10)

    assert len(losses) == 1
    backend.assert_allclose(
        parameters, backend.cast([0.0], dtype=backend.float64), atol=1e-12
    )


def _get_xxz_hamiltonian(nqubits, hamiltonian_type, backend):
    delta = 0.5
    if hamiltonian_type == "sparse":
        hamiltonian = csr_matrix(
            hamiltonians.XXZ(nqubits=nqubits, delta=delta, backend=backend).matrix
        )
        eigenvalues = scipy.sparse.linalg.eigsh(hamiltonian, k=1)
        true_gs_energy = backend.real(backend.cast(eigenvalues[0][0]))
    elif hamiltonian_type == "dense":
        hamiltonian = hamiltonians.XXZ(
            nqubits=nqubits, delta=delta, backend=backend
        ).matrix
        eigenvalues = backend.eigenvalues(hamiltonian)
        true_gs_energy = backend.real(backend.cast(eigenvalues[0]))

    return hamiltonian, true_gs_energy
