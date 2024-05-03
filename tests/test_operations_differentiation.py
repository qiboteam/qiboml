import numpy as np
import pytest
from qibo import Circuit, gates, hamiltonians

from qiboml.operations.differentiation import parameter_shift

NQUBITS = 1
GRADS_SCALEF_1 = [-0.41614683654714213, -0.29426025009181417]
GRADS_SCALEF_05 = [-0.20807341827357106, -0.14713012504590708]


def build_circuit(nqubits: int = 1):
    """Helper function to build parametric circuit."""
    circuit = Circuit(nqubits)
    for q in range(nqubits):
        circuit.add(gates.H(q))
        circuit.add(gates.RY(q, theta=0.0))
        circuit.add(gates.T(q))
        circuit.add(gates.RY(q, theta=0.0))
    circuit.add(gates.M(*range(nqubits)))
    return circuit


@pytest.mark.parametrize("nshots, atol", [(None, 1e-8), (100000, 1e-2)])
@pytest.mark.parametrize(
    "scale_factor, grads",
    [(1, GRADS_SCALEF_1), (0.5, GRADS_SCALEF_05)],
)
def test_parameter_shift(nshots, atol, scale_factor, grads):
    """Testing parameter shift rule algorithm."""

    h = hamiltonians.Z(NQUBITS)

    c = build_circuit(1)
    nparams = len(c.get_parameters())

    # testing parameter out of bounds
    with pytest.raises(ValueError):
        _ = parameter_shift(circuit=c, hamiltonian=h, parameter_index=5)

    # testing hamiltonian type
    with pytest.raises(TypeError):
        _ = parameter_shift(circuit=c, hamiltonian=c, parameter_index=0, nshots=nshots)

    params = np.linspace(0, 2, nparams, dtype=np.float64)
    c.set_parameters(params)

    psr_grads = []

    for p in range(len(params)):
        psr_grads.append(
            parameter_shift(
                hamiltonian=h,
                circuit=c,
                parameter_index=p,
                scale_factor=scale_factor,
                nshots=nshots,
            )
        )

    np.testing.assert_allclose(grads, psr_grads, atol=atol)
