import pytest
import scipy

from qiboml.models.optimizers import ExactGeodesicTransportCG
from qibo import hamiltonians
from scipy.sparse import csr_matrix


def get_xxz_hamiltonian(nqubits, hamiltonian_type, backend):
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


@pytest.mark.parametrize("nqubits", [4, 6])
@pytest.mark.parametrize("hamiltonian_type", ["sparse", "dense"])
# @pytest.mark.parametrize("hamiltonian_type", ["dense", "sparse"])
@pytest.mark.parametrize("type_loss_grad", ["exp_val", "callable"])
def test_egt_cg(
    backend,
    nqubits,
    hamiltonian_type,
    type_loss_grad,
):
    if backend.platform in ["jax", "tensorflow"] and nqubits != 4:
        pytest.skip(
            "Tests too slow with Jax and TF, will test only small size."
            + " Will test all sizes with torch."
        )

    hamiltonian, true_gs_energy = get_xxz_hamiltonian(
        nqubits, hamiltonian_type, backend
    )
    chem_acc = 0.03 / (27.2114 * backend.abs(true_gs_energy))

    if type_loss_grad == "callable":
        loss_fn = _loss_func_expval
    else:
        loss_fn = type_loss_grad

    optimizer = ExactGeodesicTransportCG(
        nqubits=nqubits,
        weight=int(nqubits / 2),
        initial_parameters=None,
        loss_fn=loss_fn,
        loss_kwargs={"hamiltonian": hamiltonian},
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