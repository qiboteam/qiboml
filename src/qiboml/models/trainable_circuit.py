from dataclasses import dataclass
from functools import cached_property
from typing import Optional

import numpy as np
from qibo.config import raise_error
from qibo.gates import Gate, ParametrizedGate
from qibo.models.circuit import Circuit, _ParametrizedGates


@dataclass
class TrainableCircuit(Circuit):

    nqubits: int
    density_matrix: bool = False
    wire_names: Optional[list] = None
    _independent_parameters_map: dict[int, set[int]] = None

    def __post_init__(self):
        super().__init__(
            self.nqubits, density_matrix=self.density_matrix, wire_names=self.wire_names
        )
        self._independent_parameters_map = {}

    @classmethod
    def from_circuit(cls, circuit: Circuit):
        obj = cls(
            nqubits=circuit.nqubits,
            density_matrix=circuit.density_matrix,
            wire_names=circuit.wire_names,
        )
        for gate in circuit.queue:
            obj.add(gate)
        return obj

    def add(self, gate: Gate):
        if isinstance(gate, ParametrizedGate):
            independent = True
            if self._independent_parameters_map != {}:
                for idx in self._independent_parameters_map:
                    indep_params = self.queue[idx].parameters[0]
                    params = gate.parameters
                    # for now only 1 parameter gates are checked
                    if len(params) == 1 and np.shares_memory(indep_params, params[0]):
                        self._independent_parameters_map[idx].add(self.ngates)
                        independent = False
                if independent:
                    self._independent_parameters_map[self.ngates] = {
                        self.ngates,
                    }
            else:
                self._independent_parameters_map = {
                    self.ngates: {
                        self.ngates,
                    }
                }
        else:
            super().add(gate)

    @property
    def independent_parameters_map(self) -> dict[int, set[int]]:
        return self._independent_parameters_map

    @independent_parameters_map.setter
    def independent_parameters_map(self, par_map: dict[int, set[int]]):
        for key, val in par_map.items():
            for idx in val:
                try:
                    gate = self.queue[idx]
                    if not isinstance(gate, ParametrizedGate):
                        raise_error(
                            RuntimeError,
                            f"Passed a parameter map containing the index ``{idx}``, corresponding to the non-parametrized gate ``{gate}``.",
                        )
                    self.queue[idx].parameters = self.queue[key].parameters
                except IndexError as e:
                    raise e
        self._independent_parameters_map = par_map

    @cached_property
    def independent_trainable_gates(self):
        gates = _ParametrizedGates()
        for idx in self.independent_parameters_map:
            if self.queue[idx].trainable:
                gates.append(self.queue[idx])
        return gates

    def get_parameters(self, independent=True):
        if independent:
            return [g.parameters for g in self.independent_trainable_gates]
        return super().get_parameters()

    def set_parameters(self, params):
        new_params = len(self.get_parameters(independent=False)) * [
            None,
        ]
        for i in range(len(self._independent_parameters_map)):
            for j in self._independent_parameters_map[i]:
                new_params[j] = params[i]
        super().set_parameters(new_params)
