import numpy as np
import pytest
import scipy
from qibo import Circuit, gates, hamiltonians
from qibo.backends import NumpyBackend
from qibo.models.encodings import _generate_rbs_angles
from qibo.quantum_info import random_statevector
from scipy.sparse import csr_matrix
from scipy.special import comb

from qiboml.models import optimizers
from qiboml.models.optimizers import ExactGeodesicTransportCG, QuantumNaturalGradient


def test_quantum_natural_gradient_uses_qfim(monkeypatch):
    backend = NumpyBackend()
    circuit = Circuit(2)
    circuit.add(gates.RY(0, theta=1.0))
    circuit.add(gates.RZ(1, theta=1.0))
    qfim_calls = []

    def loss_fn(circuit, backend):
        params = np.asarray(circuit.get_parameters(output_format="flatlist"))
        target = np.asarray([0.0, -1.0])
        return float(np.sum((params - target) ** 2))

    def gradient_fn(circuit, backend):
        params = np.asarray(circuit.get_parameters(output_format="flatlist"))
        target = np.asarray([0.0, -1.0])
        return 2 * (params - target)

    def qfim(circuit, parameters=None, **kwargs):
        qfim_calls.append(np.asarray(parameters))
        return np.diag([2.0, 4.0])

    monkeypatch.setattr(optimizers, "quantum_fisher_information_matrix", qfim)

    optimizer = QuantumNaturalGradient(
        circuit,
        loss_fn=loss_fn,
        gradient_fn=gradient_fn,
        learning_rate=0.5,
        regularization=0.0,
        backend=backend,
    )
    final_loss, losses, final_parameters = optimizer(steps=1)

    np.testing.assert_allclose(final_parameters, [0.5, 0.5])
    assert final_loss < losses[0]
    assert optimizer.n_calls_qfim == 1
    np.testing.assert_allclose(qfim_calls[0], [1.0, 1.0])


def test_quantum_natural_gradient_regularizes_singular_qfim(monkeypatch):
    backend = NumpyBackend()
    circuit = Circuit(2)
    circuit.add(gates.RY(0, theta=1.0))
    circuit.add(gates.RZ(1, theta=1.0))

    def loss_fn(circuit, backend):
        params = np.asarray(circuit.get_parameters(output_format="flatlist"))
        return float(np.sum(params**2))

    def gradient_fn(circuit, backend):
        return np.asarray([1.0, 2.0])

    monkeypatch.setattr(
        optimizers,
        "quantum_fisher_information_matrix",
        lambda *args, **kwargs: np.zeros((2, 2)),
    )

    optimizer = QuantumNaturalGradient(
        circuit,
        loss_fn=loss_fn,
        gradient_fn=gradient_fn,
        learning_rate=0.25,
        regularization=1.0,
        backend=backend,
    )
    _, _, final_parameters = optimizer(steps=1)

    np.testing.assert_allclose(final_parameters, [0.75, 0.5])


def test_quantum_natural_gradient_finite_difference_gradient(monkeypatch):
    backend = NumpyBackend()
    circuit = Circuit(1)
    circuit.add(gates.RY(0, theta=0.4))

    def loss_fn(circuit, backend):
        param = circuit.get_parameters(output_format="flatlist")[0]
        return float(param**2)

    monkeypatch.setattr(
        optimizers,
        "quantum_fisher_information_matrix",
        lambda *args, **kwargs: np.eye(1),
    )

    optimizer = QuantumNaturalGradient(
        circuit,
        loss_fn=loss_fn,
        learning_rate=0.1,
        regularization=0.0,
        finite_difference_epsilon=1e-7,
        backend=backend,
    )
    final_loss, losses, final_parameters = optimizer(steps=1)

    np.testing.assert_allclose(final_parameters, [0.32], atol=1e-6)
    assert final_loss < losses[0]
    assert optimizer.n_calls_gradient == 1


def test_quantum_natural_gradient_uses_separate_gradient_kwargs(monkeypatch):
    backend = NumpyBackend()
    circuit = Circuit(1)
    circuit.add(gates.RY(0, theta=1.0))

    def loss_fn(circuit, backend, offset):
        param = circuit.get_parameters(output_format="flatlist")[0]
        return float((param - offset) ** 2)

    def gradient_fn(circuit, backend, scale):
        param = circuit.get_parameters(output_format="flatlist")[0]
        return np.asarray([scale * param])

    monkeypatch.setattr(
        optimizers,
        "quantum_fisher_information_matrix",
        lambda *args, **kwargs: np.eye(1),
    )

    optimizer = QuantumNaturalGradient(
        circuit,
        loss_fn=loss_fn,
        loss_kwargs={"offset": 0.0},
        gradient_fn=gradient_fn,
        gradient_kwargs={"scale": 3.0},
        learning_rate=0.1,
        regularization=0.0,
        backend=backend,
    )
    final_loss, losses, final_parameters = optimizer(steps=1)

    np.testing.assert_allclose(final_parameters, [0.7])
    assert final_loss < losses[0]


def test_quantum_natural_gradient_errors():
    backend = NumpyBackend()
    circuit = Circuit(1)

    with pytest.raises(ValueError):
        _ = QuantumNaturalGradient(
            circuit, loss_fn=lambda circuit, backend: 0, backend=backend
        )

    circuit.add(gates.RY(0, theta=0.1))

    with pytest.raises(ValueError):
        _ = QuantumNaturalGradient(
            circuit,
            loss_fn=lambda circuit, backend: 0,
            learning_rate=0,
            backend=backend,
        )

    with pytest.raises(TypeError):
        _ = QuantumNaturalGradient(circuit, loss_fn=1, backend=backend)


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
