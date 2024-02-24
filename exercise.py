import time

import numpy as np
import qibo
from qibo import Circuit, gates, hamiltonians

from qiboml.ansatze import reuploading_circuit
from qiboml.differentiation import psr

qibo.set_backend("tensorflow")


Nqubits = np.arange(2, 10, 1)
nepochs = 10
nmessage = 1
nshots = None

times = []

for nqubits in Nqubits:
    print(f"nqubits: {nqubits}")
    n_times = []

    nqubits = int(nqubits)

    h = hamiltonians.Z(nqubits)
    backend = h.backend

    print("(tf, psr, tf)")
    # (tf, psr, tf)
    c = reuploading_circuit(nqubits=nqubits, nlayers=3)
    nparams = len(c.get_parameters())
    p = backend.tf.Variable(np.random.randn(nparams), dtype=backend.tf.float64)

    t1 = time.time()
    for k in range(nepochs):
        with backend.tf.GradientTape() as tape:
            c.set_parameters(p)
            exp = psr.expectation_on_backend(
                observable=h,
                circuit=c,
                # nshots=nshots,
                backend="tensorflow",
            )
        grads = tape.gradient(exp, p)
        p.assign_sub(0.05 * grads)
        if k % nmessage == 0:
            print(f"Epoch {k+1}/{nepochs}\t Exp: {exp}")
    t2 = time.time()
    exec_time = t2 - t1
    n_times.append(exec_time)

    print("(tf, psr, numpy)")

    # (tf, psr, numpy)
    c = reuploading_circuit(nqubits=nqubits, nlayers=3)
    nparams = len(c.get_parameters())
    p = backend.tf.Variable(np.random.randn(nparams), dtype=backend.tf.float64)

    t1 = time.time()
    for k in range(nepochs):
        with backend.tf.GradientTape() as tape:
            c.set_parameters(p)
            exp = psr.expectation_on_backend(
                observable=h,
                circuit=c,
                # nshots=nshots,
                backend="numpy",
            )
        grads = tape.gradient(exp, p)
        p.assign_sub(0.05 * grads)
        if k % nmessage == 0:
            print(f"Epoch {k+1}/{nepochs}\t Exp: {exp}")
    t2 = time.time()
    exec_time = t2 - t1
    n_times.append(exec_time)

    print("(tf, sim, tf)")

    # (tf, psr, numpy)
    c = reuploading_circuit(nqubits=nqubits, nlayers=3)
    nparams = len(c.get_parameters())
    p = backend.tf.Variable(np.random.randn(nparams), dtype=backend.tf.float64)

    t1 = time.time()
    for k in range(nepochs):
        with backend.tf.GradientTape() as tape:
            c.set_parameters(p)
            exp = h.expectation(c().state())
        grads = tape.gradient(exp, p)
        p.assign_sub(0.05 * grads)
        if k % nmessage == 0:
            print(f"Epoch {k+1}/{nepochs}\t Exp: {exp}")
    t2 = time.time()
    exec_time = t2 - t1
    n_times.append(exec_time)

    times.append(n_times)

np.save(arr=times, file="times")
