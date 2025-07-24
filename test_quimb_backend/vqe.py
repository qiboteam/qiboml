import numpy as np

import torch.optim as optim

from qibo import (
    Circuit, 
    gates, 
    hamiltonians, 
    set_backend, 
    construct_backend,
)

from qiboml.models.decoding import Expectation
from qiboml.interfaces.pytorch import QuantumModel
from qibo.hamiltonians.hamiltonians import SymbolicHamiltonian

# Setting number of qubits of the problem and number of layers.
nqubits = 4
nlayers = 3

# Structure of the VQE ansatz
def build_vqe_circuit(nqubits, nlayers):
    """Construct a layered, trainable ansatz."""
    c = Circuit(nqubits)
    for _ in range(nlayers):
        for q in range(nqubits):
            c.add(gates.RY(q=q, theta=np.random.randn()))
            c.add(gates.RZ(q=q, theta=np.random.randn()))
        # [c.add(gates.CRX(q0=q%nqubits, q1=(q+1)%nqubits, theta=np.random.randn())) for q in range(nqubits)]
    return c

# Define the target Hamiltonian
set_backend("qiboml", platform="pytorch")

# hamiltonian = hamiltonians.XXZ(nqubits=nqubits, delta=0.5)
from qibo.hamiltonians import SymbolicHamiltonian
from qibo.symbols import X, Z

# Define Hamiltonian using Qibo symbols
symbolic_ham = sum(Z(i) * Z(i + 1) for i in range(3))
symbolic_ham += Z(0) * Z(3)

# Define a Hamiltonian using the above form
hamiltonian = SymbolicHamiltonian(symbolic_ham)

sim_backend = construct_backend(backend="qibotn", platform="quimb")
sim_backend.configure_tn_simulation(
    ansatz="MPS",
    max_bond_dimension=10,
)
sim_backend.setup_backend_specifics(qimb_backend="torch")

# Construct the decoding layer
decoding = Expectation(
    nqubits=nqubits,
    observable=hamiltonian,
    backend=sim_backend,
    nshots=100
)

model = QuantumModel(
    circuit_structure=build_vqe_circuit(nqubits=nqubits, nlayers=nlayers),
    decoding=decoding,
)

# _ = model.draw()
# print("Exact ground state: ", min(hamiltonian.eigenvalues()))

optimizer = optim.Adam(model.parameters(), lr=0.05)

for iteration in range(300):
    optimizer.zero_grad()  
    cost = model()  
    cost.backward()  
    optimizer.step()
    print(f"Iteration {iteration}: Cost = {cost.item():.6f}")