import math
import pytest
from qiboml.models.optimizers import ExactGeodesicTransportCG 
from qibo import hamiltonians
from scipy.special import comb

def optimizer_fixture(backend):
    backend.set_seed(42)
    nqubits = 4
    weight = 2
    dim = int(comb(nqubits, weight))
    theta_init = backend.random_uniform(low=0, high=math.pi, size=dim - 1)

    hamiltonian = hamiltonians.XXZ(nqubits=nqubits, backend=backend)

    optimizer = ExactGeodesicTransportCG(
        nqubits=nqubits,
        weight=weight,
        hamiltonian=hamiltonian,
        angles=theta_init,
        backend=backend
    )
    return optimizer

def test_loss_decreases(backend):
    initial_loss = optimizer_fixture(backend).loss()
    final_loss, _, _ = optimizer_fixture(backend).run_egt_cg(steps=10)
    assert final_loss < initial_loss

def test_geometric_vs_numerical(backend):
    backend.set_seed(42)
    nqubits = 4
    weight = 2

    dim = int(comb(nqubits, weight))
    theta_init = backend.random_uniform(low=0, high=math.pi, size=dim - 1)

    hamiltonian = hamiltonians.XXZ(nqubits=nqubits, backend=backend)

    # Optimizer with numerical gradient
    optimizer_num = ExactGeodesicTransportCG(
        nqubits=nqubits,
        weight=weight,
        hamiltonian=hamiltonian,
        angles=theta_init.copy(),
        geometric_gradient=False,
        backend=backend,
    )
    final_loss_num, _ ,_ = optimizer_num.run_egt_cg(steps=10)

    # Optimizer with geometric gradient
    optimizer_geo = ExactGeodesicTransportCG(
        nqubits=nqubits,
        weight=weight,
        hamiltonian=hamiltonian,
        angles=theta_init.copy(),
        geometric_gradient=True,
        backend=backend,
    )
    final_loss_geo, _, _ = optimizer_geo(steps=10)

    assert backend.assert_allclose(final_loss_geo, final_loss_num, rtol=0.05)

