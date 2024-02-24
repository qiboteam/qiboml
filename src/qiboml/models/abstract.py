from abc import ABC, abstractmethod
from typing import Tuple

from qibo import Circuit, gates
from qibo.config import raise_error


class ReuploadingCircuit(ABC):
    def __init__(self, nqubits: int, nlayers: int, data_dimensionality: Tuple):
        """
        Abstract Data Reuploading Circuit.

        Args:
            nqubits (int): number of qubits;
            nlayers (int): number of layers;
            data_dimensionality (Tuple): data dimensionality.
        """
        self.nqubits = nqubits
        self.nlayers = nlayers
        self.datadim = data_dimensionality
        self.circuit = Circuit(nqubits)
        self.parameters = self.circuit.get_parameters()

    @property
    def nparams(self):
        """Number of trainable parameters."""
        return len(self.parameters)

    @abstractmethod
    def build_circuit(self):
        """Build the Parametric Circuit according to the chosen strategy."""
        raise_error(NotImplementedError)

    def build_entangling_layer(self):
        """Build circuit's entangling layer structure."""
        c = Circuit(self.nqubits)
        for q in range(1, self.nqubits - 1, 1):
            c.add(gates.CNOT(q0=q, q1=q + 1))
        c.add(gates.CNOT(q0=self.nqubits - 1, q1=0))
        return c

    def set_parameters(self, parameters):
        """Set trainable parameters into the circuit."""
        self.circuit.set_parameters(parameters)

    @abstractmethod
    def inject_data(self, x):
        """Inject data ``x`` into the circuit according to the chosen strategy."""
        raise_error(NotImplementedError)
