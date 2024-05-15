import os
import time

import qibo.backends

# disabling hardware accelerators warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import matplotlib.pyplot as plt
import numpy as np
import qibo
import tensorflow as tf
from qibo import Circuit, gates, hamiltonians, set_backend

from qiboml.operations import differentiation, expectation

set_backend("numpy")

# -------------------- Helpers -------------------------------------------------


def build_parametric_circuit(nqubits, nlayers):
    """Build a Parametric Quantum Circuit with Qibo."""

    c = Circuit(nqubits)
    for _ in range(nlayers):
        for q in range(nqubits):
            c.add(gates.RY(q=q, theta=0))
            c.add(gates.RZ(q=q, theta=0))
        for q in range(0, nqubits - 1, 1):
            c.add(gates.CNOT(q0=q, q1=q + 1))
        c.add(gates.CNOT(q0=nqubits - 1, q1=0))
    c.add(gates.M(*range(nqubits)))

    return c


def train_circuit(
    circuit, hamiltonian, nepochs, exec_backend, differentiation_rule=None, nshots=None
):
    """
    Perform a simple gradient descent training of ``circuit`` to minimize the expectation
    value of ``hamiltonian``. Gradients are computed via the chosen ``differentiation_rule``
    and expectation values calculated executing circuit on the selected ``exec_backend``.

    Returns:
        float: total execution time.
    """
    # a couple of hyper-parameters
    learning_rate = 0.01
    random_seed = 42

    # random parameters
    np.random.seed(random_seed)
    nparams = len(circuit.get_parameters())
    params = tf.Variable(np.random.uniform(0, 2 * np.pi, nparams))

    it = time.time()

    for epoch in range(nepochs):
        with tf.GradientTape() as tape:
            tape.watch(params)
            circuit.set_parameters(params)
            cost = expectation.expectation(
                observable=hamiltonian,
                circuit=circuit,
                exec_backend=exec_backend,
                differentiation_rule=differentiation_rule,
                nshots=nshots,
            )
            if epoch % 10 == 0:
                print(f"Cost: {round(cost, 4)} \t |\t Epoch: {epoch}")
            gradients = tape.gradient(cost, params)
            params = params.assign_sub(learning_rate * gradients)
    ft = time.time()

    return ft - it


# ---------------- variables initialization ------------------------------------

tf_backend = qibo.backends.construct_backend("tensorflow")
np_backend = qibo.backends.construct_backend("numpy")

nqubits = 7
nlayers = 7

# setup the problem
circuit = build_parametric_circuit(nqubits, nlayers)
hamiltonian = hamiltonians.Z(nqubits)

train_time = train_circuit(
    circuit=circuit,
    hamiltonian=hamiltonian,
    nepochs=10,
    exec_backend=tf_backend,
    differentiation_rule=differentiation.symbolical,
)

print(f"Execution time: {train_time}")
