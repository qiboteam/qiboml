import pytest
import scipy

from qiboml.models.optimizers import ExactGeodesicTransportCG
from qibo import hamiltonians, set_backend

from qibo.quantum_info import random_statevector
from qibo.models.encodings import _generate_rbs_angles

from scipy.sparse import csr_matrix
from scipy.special import comb


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
    with pytest.raises(TypeError):
        # if loss_fn is a callable, we must use a backend that has autodiff
        loss_fn = _loss_func_expval
        backend = set_backend("numpy")
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
            backend.real(
                random_statevector(
                    int(comb(nqubits, int(nqubits / 2))), seed=13, backend=backend
                )
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
    """
    Backend-agnostic expectation value <psi|H|psi>.

    Supports:
    - NumPy / SciPy sparse
    - JAX (BCOO)
    - TensorFlow (tf.sparse.SparseTensor)
    - PyTorch (sparse COO / CSR)

    Assumes hamiltonian is sparse in the backend's native format
    """
    psi = backend.execute_circuit(circuit).state()
    platform = backend.platform
    if platform == "tensorflow":
        psi_col = backend.engine.reshape(psi, (-1, 1))
        h_psi = backend.engine.sparse.sparse_dense_matmul(hamiltonian, psi_col)
        h_psi = backend.engine.reshape(h_psi, (-1,))
    elif platform == "pytorch":
        h_psi = backend.engine.sparse.mm(hamiltonian, psi.unsqueeze(1)).squeeze(1)
    else:
        h_psi = hamiltonian @ psi
    return backend.real(backend.sum(backend.conj(psi) * h_psi))


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
