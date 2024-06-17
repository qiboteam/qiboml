import os
import time

# disabling hardware accelerators warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import jax
import jax.numpy as jnp
import numpy as np
import qibo
import tensorflow as tf
from qibo import Circuit, gates, hamiltonians
from qibo.backends import construct_backend

from qiboml.backends import JaxBackend, TensorflowBackend
from qiboml.operations import differentiation, expectation


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

    def cost_function(params):
        circuit.set_parameters(params)
        expval = expectation.expectation(
            observable=hamiltonian,
            circuit=circuit,
            exec_backend=exec_backend,
            differentiation_rule=differentiation_rule,
            nshots=nshots,
        )
        return expval

    # random parameters
    np.random.seed(random_seed)
    nparams = len(circuit.get_parameters())
    params = np.random.uniform(0, 2 * np.pi, nparams)

    it = time.time()

    if isinstance(hamiltonian.backend, TensorflowBackend):
        params = tf.Variable(params)
        for epoch in range(nepochs):
            with tf.GradientTape() as tape:
                cost = cost_function(params)
                if epoch % 2 == 0:
                    print(f"Cost: {round(cost, 4)} \t |\t Epoch: {epoch}")
                gradients = tape.gradient(cost, params)
                init_params = params.assign_sub(learning_rate * gradients)

    if isinstance(hamiltonian.backend, JaxBackend):
        jitted_cost_function = jax.jit(cost_function)
        dcost = jax.grad(cost_function)
        for epoch in range(nepochs):
            gradients = dcost(params)
            cost = cost_function(params)
            if epoch % 2 == 0:
                print(f"Cost: {cost:.4} \t |\t Epoch: {epoch}")
            params -= learning_rate * gradients

    ft = time.time()
    return ft - it


# frontend definition
qibo.set_backend("tensorflow")
# execution backend
backend = construct_backend("numpy")

nqubits = 3
c = build_parametric_circuit(nqubits, 2)
h = hamiltonians.Z(nqubits)

extime = train_circuit(
    circuit=c,
    hamiltonian=h,
    nepochs=20,
    exec_backend=backend,
    differentiation_rule=differentiation.parameter_shift,
)

print(f"Total execution time: {extime}")
