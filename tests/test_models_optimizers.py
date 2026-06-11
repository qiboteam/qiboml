from types import MethodType

import numpy as np
import pytest
import scipy
from qibo import Circuit, gates, hamiltonians
from qibo.backends import NumpyBackend
from qibo.models.encodings import _generate_rbs_angles
from qibo.quantum_info import quantum_fisher_information_matrix, random_statevector
from scipy.sparse import csr_matrix
from scipy.special import comb

from qiboml.models.optimizers import ExactGeodesicTransportCG, QuantumNaturalGradient


def _qibo_qfim(circuit, parameters, backend):
    original_cast = backend.cast
    original_jacobian = getattr(backend, "jacobian", None)

    def cast_with_numpy_dtypes(bound_backend, array, dtype=None, copy=False, **kwargs):
        if dtype is not None:
            try:
                dtype = getattr(bound_backend, np.dtype(dtype).name)
            except (AttributeError, TypeError):
                pass
        return original_cast(array, dtype=dtype, copy=copy, **kwargs)

    def jax_jacobian(
        bound_backend, copied_circuit, params, initial_state=None, return_complex=True
    ):
        copied = copied_circuit.copy(deep=True)

        def state_components(local_params):
            copied.set_parameters(local_params)
            state = bound_backend.execute_circuit(
                copied, initial_state=initial_state
            ).state()
            if return_complex:
                return bound_backend.real(state), bound_backend.imag(state)
            return bound_backend.real(state)

        return bound_backend.jax.jacobian(state_components)(params)

    backend.cast = MethodType(cast_with_numpy_dtypes, backend)
    if backend.platform == "jax":
        backend.jacobian = MethodType(jax_jacobian, backend)

    try:
        return quantum_fisher_information_matrix(
            circuit,
            parameters=parameters,
            backend=backend,
        )
    finally:
        backend.cast = original_cast
        if backend.platform == "jax":
            backend.jacobian = original_jacobian
        elif original_jacobian is not None:
            backend.jacobian = original_jacobian


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
        callback=None,
        seed=13,
        backend=backend,
    )
    _, losses, _ = optimizer(steps=20)
    rel_errors = backend.abs(1 - losses / true_gs_energy)

    assert rel_errors[-1] < chem_acc


def _loss_func_expval(circuit, backend, *, hamiltonian):
    psi = backend.execute_circuit(circuit).state()
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
        callback=None,
        seed=13,
        backend=backend,
    )
    _, losses, _ = optimizer(steps=20)
    rel_errors = backend.abs(1 - losses / true_gs_energy)

    assert rel_errors[-1] < chem_acc


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


def _build_ry_cz_ansatz(nqubits):
    circuit = Circuit(nqubits)
    for qubit in range(nqubits):
        circuit.add(gates.RY(qubit, theta=0.0))
    for qubit in range(nqubits - 1):
        circuit.add(gates.CZ(qubit, qubit + 1))
    for qubit in range(nqubits):
        circuit.add(gates.RY(qubit, theta=0.0))
    return circuit


@pytest.mark.parametrize("nqubits", [2, 3])
def test_quantum_natural_gradient_vqe(backend, nqubits):
    circuit = _build_ry_cz_ansatz(nqubits)
    hamiltonian = hamiltonians.Z(nqubits, backend=backend).matrix
    initial_parameters = backend.cast(
        np.linspace(0.1, 0.6, len(circuit.get_parameters())),
        dtype=backend.float64,
    )

    optimizer = QuantumNaturalGradient(
        circuit=circuit,
        loss_fn="exp_val",
        loss_kwargs={"hamiltonian": hamiltonian},
        initial_parameters=initial_parameters,
        learning_rate=0.2,
        regularization=1e-3,
        backend=backend,
    )
    final_loss, losses, final_parameters = optimizer(steps=40, tolerance=1e-10)
    circuit_parameters = backend.cast(
        np.asarray(
            backend.to_numpy(circuit.get_parameters()), dtype=np.float64
        ).reshape(-1),
        dtype=backend.float64,
    )

    assert optimizer.n_calls_loss >= len(losses)
    assert optimizer.n_calls_gradient >= 1
    backend.assert_allclose(final_parameters, circuit_parameters, atol=1e-7)
    assert backend.abs(final_loss + nqubits) < 1e-5


def test_quantum_natural_gradient_metric_tensor(backend):
    circuit = _build_ry_cz_ansatz(3)
    parameters = backend.cast(
        [0.17, -0.31, 0.29, 0.12, -0.44, 0.08], dtype=backend.float64
    )
    circuit.set_parameters(parameters)

    optimizer = QuantumNaturalGradient(
        circuit=circuit,
        loss_fn="exp_val",
        loss_kwargs={"hamiltonian": hamiltonians.Z(3, backend=backend).matrix},
        backend=backend,
    )
    metric = optimizer.metric_tensor()
    expected = backend.real(_qibo_qfim(circuit, parameters, backend)) / 4.0
    metric_np = np.asarray(backend.to_numpy(metric), dtype=np.float64)
    metric_transpose = metric_np.T

    np.testing.assert_allclose(
        metric_np,
        np.asarray(backend.to_numpy(expected), dtype=np.float64),
        atol=1e-7,
    )
    np.testing.assert_allclose(metric_np, metric_transpose, atol=1e-7)


def test_quantum_natural_gradient_callable_loss(backend):
    circuit = _build_ry_cz_ansatz(2)
    initial_parameters = backend.cast([0.2, -0.1, 0.3, 0.4], dtype=backend.float64)
    circuit.set_parameters(initial_parameters)

    def callable_loss(circuit, backend):
        state = backend.execute_circuit(circuit).state()
        return 1.0 - backend.real(backend.conj(state[0]) * state[0])

    initial_loss = np.asarray(backend.to_numpy(callable_loss(circuit, backend))).item()

    optimizer = QuantumNaturalGradient(
        circuit=circuit,
        loss_fn=callable_loss,
        initial_parameters=initial_parameters,
        learning_rate=0.05,
        regularization=1e-3,
        backend=backend,
    )
    loss = optimizer.step()
    circuit_parameters = backend.cast(
        np.asarray(
            backend.to_numpy(circuit.get_parameters()), dtype=np.float64
        ).reshape(-1),
        dtype=backend.float64,
    )

    assert optimizer.circuit is circuit
    assert optimizer.n_calls_gradient == 1
    assert optimizer.n_calls_loss >= 1
    backend.assert_allclose(circuit_parameters, optimizer.parameters, atol=1e-7)
    assert np.asarray(backend.to_numpy(loss)).item() <= initial_loss


def test_quantum_natural_gradient_callable_loss_errors():
    backend = NumpyBackend()
    circuit = _build_ry_cz_ansatz(2)

    def callable_loss(circuit, backend):
        state = backend.execute_circuit(circuit).state()
        return 1.0 - backend.real(backend.conj(state[0]) * state[0])

    with pytest.raises(TypeError):
        _ = QuantumNaturalGradient(
            circuit=circuit,
            loss_fn=callable_loss,
            backend=backend,
        )


def test_quantum_natural_gradient_errors(backend):
    circuit = _build_ry_cz_ansatz(2)

    with pytest.raises(TypeError):
        _ = QuantumNaturalGradient(
            circuit=circuit,
            loss_fn=13,
            loss_kwargs={"hamiltonian": hamiltonians.Z(2, backend=backend).matrix},
            backend=backend,
        )

    with pytest.raises(ValueError):
        _ = QuantumNaturalGradient(
            circuit=circuit,
            loss_fn="expval",
            loss_kwargs={"hamiltonian": hamiltonians.Z(2, backend=backend).matrix},
            backend=backend,
        )

    with pytest.raises(TypeError):
        _ = QuantumNaturalGradient(
            circuit=circuit,
            loss_fn="exp_val",
            loss_kwargs={"hamiltonian": [("bad", "hamiltonian")]},
            backend=backend,
        )

    with pytest.raises(ValueError):
        _ = QuantumNaturalGradient(
            circuit=circuit,
            loss_fn="exp_val",
            loss_kwargs={},
            backend=backend,
        )


def test_quantum_natural_gradient_sparse_hamiltonian_callback_and_tolerance(
    backend,
):
    circuit = _build_ry_cz_ansatz(2)
    hamiltonian = csr_matrix(hamiltonians.Z(2, backend=backend).matrix)
    callback_calls = []

    optimizer = QuantumNaturalGradient(
        circuit=circuit,
        loss_fn="exp_val",
        loss_kwargs={"hamiltonian": hamiltonian},
        initial_parameters=backend.cast([0.2, -0.1, 0.3, 0.4], dtype=backend.float64),
        callback=lambda **kwargs: callback_calls.append(kwargs),
        backend=backend,
    )
    optimizer.gradient = lambda: backend.cast(
        np.zeros(len(optimizer.parameters)),
        dtype=backend.float64,
    )

    final_loss, losses, _ = optimizer.run_qng(steps=5, tolerance=1e-12)

    assert len(losses) == 1
    assert len(callback_calls) == 1
    assert callback_calls[0]["iter_num"] == 1
    backend.assert_allclose(callback_calls[0]["loss"], losses[0], atol=1e-7)
    backend.assert_allclose(
        callback_calls[0]["parameters"], optimizer.parameters, atol=1e-7
    )
    assert optimizer.n_calls_loss == 1
    backend.assert_allclose(final_loss, losses[0], atol=1e-7)
