import os
import time

# disabling hardware accelerators warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import qibo
import tensorflow as tf
import torch
from qibo import Circuit, gates, hamiltonians
from qibo.backends import construct_backend

from qiboml.backends import JaxBackend, PyTorchBackend, TensorflowBackend
from qiboml.operations import differentiation, expectation


def build_parametric_circuit(nqubits, nlayers):
    """Build a Parametric Quantum Circuit with Qibo."""
    c = Circuit(nqubits)
    for _ in range(nlayers):
        for q in range(nqubits):
            c.add(gates.RY(q=q, theta=0.0))
            c.add(gates.RZ(q=q, theta=0.0))
        for q in range(0, nqubits - 1, 1):
            c.add(gates.CNOT(q0=q, q1=q + 1))
        c.add(gates.CNOT(q0=nqubits - 1, q1=0))
    c.add(gates.M(*range(nqubits)))
    return c


def train_circuit(
    circuit, hamiltonian, nepochs, exec_backend, differentiation_rule=None, nshots=None
):
    learning_rate = 0.01
    random_seed = 42

    def cost_function(params):
        expval = expectation.expectation(
            parameters=params,
            observable=hamiltonian,
            circuit=circuit,
            exec_backend=exec_backend,
            differentiation_rule=differentiation_rule,
            nshots=nshots,
        )
        return expval

    np.random.seed(random_seed)
    nparams = len(circuit.get_parameters())
    params = np.random.uniform(0, 2 * np.pi, nparams)
    circuit.set_parameters(params)

    it = time.time()

    if isinstance(hamiltonian.backend, TensorflowBackend):
        init_params = tf.Variable(params)
        for epoch in range(nepochs):
            with tf.GradientTape() as tape:
                cost = cost_function(init_params)
                if epoch % 2 == 0:
                    print(f"Cost: {round(cost, 4)} \t |\t Epoch: {epoch}")
                gradients = tape.gradient(cost, init_params)
                init_params.assign_sub(learning_rate * gradients)

    elif isinstance(hamiltonian.backend, PyTorchBackend):
        params = torch.tensor(params, dtype=torch.float64, requires_grad=True)
        optimizer = torch.optim.SGD([params], lr=learning_rate)
        for epoch in range(nepochs):
            optimizer.zero_grad()
            cost = cost_function(params)
            cost.backward()
            optimizer.step()
            if epoch % 2 == 0:
                print(f"Cost: {cost.item():.4} \t |\t Epoch: {epoch}")

    ft = time.time()
    return ft - it


# frontend definition
qibo.set_backend("pytorch")
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
    nshots=1000,
    differentiation_rule=differentiation.parameter_shift,
)

print(f"Total execution time: {extime}")
