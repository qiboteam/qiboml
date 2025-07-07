import numpy as np
import pytest
from qiboml.models.optimizers import ExactGeodesicTransportCG 
from qibo import hamiltonians
from scipy.special import comb

@pytest.fixture
def optimizer_fixture():
    np.random.seed(42)
    n_qubits = 4
    weight = 2
    dim = int(comb(n_qubits, weight))
    theta_init = np.random.uniform(low=0, high=np.pi, size=dim - 1)

    hamiltonian = hamiltonians.XXZ(nqubits=n_qubits)

    optimizer = ExactGeodesicTransportCG(
        n_qubits=n_qubits,
        weight=weight,
        hamiltonian=hamiltonian,
        theta=theta_init,
    )
    return optimizer

def test_loss_decreases(optimizer_fixture):
    initial_loss = optimizer_fixture.loss()
    final_loss, _, _ = optimizer_fixture.run_egt_cg(steps=10)
    assert final_loss < initial_loss

def test_geometric_vs_numerical():
    np.random.seed(42)
    n_qubits = 4
    weight = 2

    dim = int(comb(n_qubits, weight))
    theta_init = np.random.uniform(low=0, high=np.pi, size=dim - 1)

    hamiltonian = hamiltonians.XXZ(nqubits=n_qubits)

    # Optimizer with numerical gradient
    optimizer_num = ExactGeodesicTransportCG(
        n_qubits=n_qubits,
        weight=weight,
        hamiltonian=hamiltonian,
        theta=theta_init.copy(),
        geometric_gradient=False,
    )
    final_loss_num, _ ,_ = optimizer_num.run_egt_cg(steps=10)

    # Optimizer with geometric gradient
    optimizer_geo = ExactGeodesicTransportCG(
        n_qubits=n_qubits,
        weight=weight,
        hamiltonian=hamiltonian,
        theta=theta_init.copy(),
        geometric_gradient=True,
    )
    final_loss_geo, _, _ = optimizer_geo.run_egt_cg(steps=10)

    assert final_loss_geo == pytest.approx(final_loss_num, rel=0.05)



