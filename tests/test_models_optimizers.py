import math

import pytest
import scipy
from qibo import Circuit, gates, hamiltonians
from qibo.backends import NumpyBackend
from qibo.models.encodings import _generate_rbs_angles
from qibo.quantum_info import random_statevector
from scipy.sparse import csr_matrix
from scipy.special import comb

from qiboml.models.optimizers import ExactGeodesicTransportCG, QuantumNaturalGradient


def _z_expectation(circuit, backend):
    state = backend.execute_circuit(circuit).state()
    pauli_z = backend.cast([[1, 0], [0, -1]], dtype=backend.complex128)
    return backend.real(backend.sum(backend.conj(state) * (pauli_z @ state)))


def test_quantum_natural_gradient(backend):
    circuit = Circuit(1)
    circuit.add(gates.RY(0, theta=0.4))
    callback_steps = []

    optimizer = QuantumNaturalGradient(
        circuit=circuit,
        loss_fn=_z_expectation,
        learning_rate=0.2,
        regularization=1e-8,
        callback=lambda iter_num, **_: callback_steps.append(iter_num),
        backend=backend,
    )
    final_loss, losses, final_parameters = optimizer(steps=30)

    assert final_loss < -0.99
    assert losses[-1] < losses[0]
    assert final_parameters.shape == (1,)
    assert callback_steps == list(range(1, len(callback_steps) + 1))
    assert 0 < len(callback_steps) <= 30


def test_quantum_natural_gradient_step_matches_expected_update(backend):
    circuit = Circuit(1)
    circuit.add(gates.RY(0, theta=0.4))
    learning_rate = 0.1
    optimizer = QuantumNaturalGradient(
        circuit=circuit,
        loss_fn=_z_expectation,
        learning_rate=learning_rate,
        regularization=0.0,
        backend=backend,
    )

    _, gradient, natural_gradient, metric = optimizer.step()

    backend.assert_allclose(
        metric, backend.cast([[0.25]], dtype=backend.float64), atol=1e-7
    )
    backend.assert_allclose(natural_gradient, 4.0 * gradient, atol=1e-7)
    backend.assert_allclose(
        optimizer.parameters,
        backend.cast(
            [0.4 + 4.0 * learning_rate * math.sin(0.4)],
            dtype=backend.float64,
        ),
        atol=1e-7,
    )


@pytest.mark.parametrize("nqubits", [2, 3])
def test_quantum_natural_gradient_vqe(backend, nqubits):
    circuit = Circuit(nqubits)
    for qubit in range(nqubits):
        circuit.add(gates.RY(qubit, theta=0.1 + 0.2 * qubit))
    for qubit in range(nqubits - 1):
        circuit.add(gates.CZ(qubit, qubit + 1))

    optimizer = QuantumNaturalGradient(
        circuit=circuit,
        loss_fn="exp_val",
        loss_kwargs={"hamiltonian": hamiltonians.Z(nqubits, backend=backend).matrix},
        learning_rate=0.2,
        backend=backend,
    )

    final_loss, losses, final_parameters = optimizer.run_qng(steps=25)

    assert final_loss < -nqubits + 1e-7
    assert losses[-1] < losses[0]
    backend.assert_allclose(
        final_parameters,
        backend.cast(circuit.get_parameters(), dtype=backend.float64).flatten(),
        atol=1e-7,
    )


def test_quantum_natural_gradient_errors(backend):
    circuit = Circuit(1)
    circuit.add(gates.RY(0, theta=0.4))

    with pytest.raises(TypeError):
        QuantumNaturalGradient(circuit, _z_expectation, backend=NumpyBackend())

    with pytest.raises(ValueError):
        QuantumNaturalGradient(
            circuit, _z_expectation, learning_rate=0, backend=backend
        )
    with pytest.raises(ValueError):
        QuantumNaturalGradient(
            circuit, _z_expectation, regularization=-1, backend=backend
        )
    with pytest.raises(ValueError):
        QuantumNaturalGradient(circuit, "exp_val", backend=backend)


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
