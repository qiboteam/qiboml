from math import exp, log
from typing import Callable, Tuple

from qibo import Circuit, gates

from qiboml.models.abstract import ReuploadingCircuit


class ReuploadingU3(ReuploadingCircuit):
    def __init__(
        self,
        nqubits: int,
        nlayers: int,
        data_dimensionality: Tuple,
        actf1: Callable = lambda x: x,
        actf2: Callable = lambda x: log(x),
        actf3: Callable = lambda x: exp(x),
    ):
        """Reuplading U3 ansatz."""

        super().__init__(nqubits, nlayers, data_dimensionality)
        self.actf1 = actf1
        self.actf2 = actf2
        self.actf3 = actf3
        self.circuit = self.build_circuit()

    def build_circuit(self):
        c = Circuit(self.nqubits)
        for _ in range(self.nlayers):
            for q in range(self.nqubits):
                c.add(gates.U3(q=q, theta=0, phi=0, lam=0))
            if self.nqubits >= 2:
                c += self.build_entangling_layer()
        c.add(gates.M(*range(self.nqubits)))
        return c

    def inject_data(self, x):
        new_parameters = []
        k = 0
        for _ in range(self.nlayers):
            for _ in range(self.nqubits):
                new_parameters.append(
                    self.parameters[k] * self.actf1(x) + self.parameters[k + 1]
                )
                new_parameters.append(
                    self.parameters[k + 2] * self.actf2(x) + self.parameters[k + 3]
                )
                new_parameters.append(
                    self.parameters[k + 4] * self.actf3(x) + self.parameters[k + 5]
                )
            k += 6
        self.circuit.set_parameters(new_parameters)
