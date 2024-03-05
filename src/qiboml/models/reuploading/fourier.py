from typing import Tuple

import numpy as np
from qibo import Circuit, gates

from qiboml.models.abstract import ReuploadingCircuit


class FourierReuploading(ReuploadingCircuit):
    def __init__(self, nqubits: int, nlayers: int, data_dimensionality: Tuple):
        """Reuplading U3 ansatz."""

        super().__init__(nqubits, nlayers, data_dimensionality)
        self.circuit = self.build_circuit()
        self.parameters = self._init_parameters()

    def _init_parameters(self):
        """
        Initialize parameters using random numbers.
        In FourierReuploading we use 2/3 of the rotational gates as trainable gates.
        """
        return np.random.randn(
            int(2 / 3 * len(self.circuit.get_parameters(format="flatlist")))
        )

    def build_circuit(self):
        c = Circuit(self.nqubits)
        for _ in range(self.nlayers):
            self.build_encoding_layer()
            if self.nqubits >= 2:
                c += self.build_entangling_layer()
            self.build_training_layer()
        c.add(gates.M(*range(self.nqubits)))
        return c

    def build_training_layer(self):
        c = Circuit(self.nqubits)
        for q in range(self.nqubits):
            c.add(gates.RY(q=q, theta=0))
            c.add(gates.RZ(q=q, theta=0))
        return c

    def build_encoding_layer(self):
        c = Circuit(self.nqubits)
        for q in range(self.nqubits):
            c.add(gates.RX(q=q, theta=0))
        return c

    def inject_data(self, x):
        new_parameters = []
        k = 0
        for _ in range(self.nlayers):
            for _ in range(self.nqubits):
                new_parameters.append(self.parameters[k])
                new_parameters.append(self.parameters[k + 1])
                new_parameters.append(x)
            k += 2
        self.circuit.set_parameters(new_parameters)
