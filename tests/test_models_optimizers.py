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
def test_egt_cg(
    backend,
    nqubits,
    hamiltonian_type,
):
    hamiltonian, true_gs_energy = get_xxz_hamiltonian(
        nqubits, hamiltonian_type, backend
    )
    print(f"backend: {backend}\nhamilt type: {type(hamiltonian)}")
    chem_acc = 0.03 / (27.2114 * backend.abs(true_gs_energy))

    optimizer = ExactGeodesicTransportCG(
        nqubits=nqubits,
        weight=int(nqubits / 2),
        initial_parameters=None,
        loss_fn="exp_val",
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
