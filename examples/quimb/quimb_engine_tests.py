import numpy as np
import qibo
from qibo import Circuit, gates, hamiltonians
from qibo.backends import construct_backend
np.random.seed(42)

nqubits = 2
quimb_backend = construct_backend(backend="qibotn", platform="quimb")
def build_circuit(nqubits, nlayers):
    """Construct a parametric quantum circuit."""
    circ = Circuit(nqubits)
    for _ in range(nlayers):
        for q in range(nqubits):
            circ.add(gates.RY(q=q, theta=0.0))
            circ.add(gates.RZ(q=q, theta=0.0))
        [circ.add(gates.CNOT(q % nqubits, (q + 1) % nqubits) for q in range(nqubits))]
    circ.add(gates.M(*range(nqubits)))
    return circ
circuit = build_circuit(nqubits=nqubits, nlayers=3)
circuit.set_parameters(
    parameters=np.random.uniform(-np.pi, np.pi, len(circuit.get_parameters())),
)

quimb_backend.configure_tn_simulation(
    ansatz="MPS",
    max_bond_dimension=10,
)
quimb_backend.setup_backend_specifics(qimb_backend="torch")
outcome = quimb_backend.execute_circuit(circuit=circuit, nshots=2000, return_array=True)

print("Number of qubits: ", circuit.nqubits)
print("Probabilities", outcome.probabilities())
print("Frequencies", outcome.frequencies())
