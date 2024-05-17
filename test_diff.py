import time

import qibo
from qibo import Circuit, gates, hamiltonians
from qibo.backends import construct_backend

from qiboml.operations.differentiation import (
    parameter_shift,
    symbolical,
    symbolical_with_jax,
)


def build_parametric_circuit(nqubits, nlayers):
    """Build a Parametric Quantum Circuit with Qibo."""
    c = Circuit(nqubits)
    for _ in range(nlayers):
        for q in range(nqubits):
            c.add(gates.RY(q=q, theta=0.3))
            c.add(gates.RZ(q=q, theta=0.2))
        for q in range(0, nqubits - 1, 1):
            c.add(gates.CNOT(q0=q, q1=q + 1))
        c.add(gates.CNOT(q0=nqubits - 1, q1=0))
    c.add(gates.M(*range(nqubits)))
    return c


def compute_derivatives(
    nqubits, nlayers, nshots, differentiation_rule, frontend, backend
):
    """Compute derivatives of the expectation of an nqubits Z w.r.t. PQC params."""

    qibo.set_backend(frontend)
    exec_backend = construct_backend(backend)

    c = build_parametric_circuit(nqubits, nlayers)
    h = hamiltonians.Z(nqubits)

    kwargs = dict(
        hamiltonian=h,
        circuit=c,
        exec_backend=exec_backend,
    )

    if nshots is not None:
        kwargs.update({"nshots": nshots})

    it = time.time()
    grads = differentiation_rule(**kwargs)
    exectime = time.time() - it

    return grads, exectime


print(
    compute_derivatives(
        nqubits=6,
        nlayers=6,
        differentiation_rule=symbolical_with_jax,
        nshots=None,
        frontend="tensorflow",
        backend="jax",
    )
)
