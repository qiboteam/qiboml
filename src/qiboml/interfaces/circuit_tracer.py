from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property
from inspect import signature
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
from qibo import Circuit

from qiboml import ndarray
from qiboml.models.encoding import QuantumEncoding


@dataclass
class CircuitTracer(ABC):

    circuit_structure: List[Union[Circuit, QuantumEncoding, Callable]]
    derivation_mode: str = "forward"

    @property
    @abstractmethod
    def engine(self):
        pass

    @staticmethod
    @abstractmethod
    def jacfwd(f: Callable, argnums: Union[int, Tuple[int]]):
        pass

    @staticmethod
    @abstractmethod
    def jacrev(f: Callable, argnums: Union[int, Tuple[int]]):
        pass

    def _compute_jacobian_functional(
        self, circuit: Union[QuantumEncoding, Callable]
    ) -> Callable:

        _tmp_circuit = circuit

        def build(x):
            is_encoding = isinstance(_tmp_circuit, QuantumEncoding)
            circuit = _tmp_circuit(x) if is_encoding else _tmp_circuit(*x)
            return self.engine.vstack(
                [
                    p
                    for pars in circuit.get_parameters(
                        include_not_trainable=is_encoding
                    )
                    for p in pars
                ]
            )

        jac = self.jacfwd if self.derivation_mode == "forward" else self.jacrev
        return jac(build, argnums=0)

    @cached_property
    def jacobian_functionals(
        self,
    ) -> dict[int, Callable]:
        jacobians = {}
        jac = self.jacfwd if self.derivation_mode == "forward" else self.jacrev
        for circ in self.circuit_structure:
            if isinstance(circ, Circuit):
                jacobians[id(circ)] = self.identity
            else:
                jacobians[id(circ)] = self._compute_jacobian_functional(
                    circ
                )  # jac(build, argnums=0)
        return jacobians

    def trace(self, f: Callable, params: ndarray):
        # we always assume the input is a 1-dim array, even for encodings
        # thus the jacobian is always a matrix
        jac = self.engine.reshape(
            self.jacobian_functionals[id(f)](params), (-1, len(params))
        )
        par_map = {}
        for i, row in enumerate(jac):
            for j in self.engine.nonzero(row):
                j = int(j)
                if j in par_map:
                    par_map[j] += (i,)
                else:
                    par_map[j] = (i,)
        return jac, par_map

    def _build_from_encoding(
        self, encoding: QuantumEncoding, x: ndarray, trace: bool = True
    ):
        circuit = encoding(x)
        if trace:
            return *self.trace(encoding, x), circuit
        return circuit

    def _build_from_circuit(
        self, circuit: Circuit, params: ndarray, trace: bool = True
    ):
        circuit.set_parameters(params)
        if trace:
            jacobian = self.identity(
                len(params),
                dtype=self._get_dtype(params),
                device=self._get_device(params),
            )
            # all the circuit parameters are considered independent
            par_map = {i: (i,) for i in enumerate(params)}
            return jacobian, par_map, circuit
        return circuit

    def _build_from_callable(self, f: Callable, params: ndarray, trace: bool = True):
        circuit = f(*params)
        if trace:
            return *self.trace(f, params), circuit
        return circuit

    @staticmethod
    def _get_device(array: ndarray):
        return array.device

    @staticmethod
    def _get_dtype(array: ndarray):
        return array.dtype

    @abstractmethod
    def identity(self, dim: int, dtype, device) -> ndarray:
        pass

    @abstractmethod
    def zeros(self, shape: Union[int, Tuple[int]], dtype, device) -> ndarray:
        pass

    def __call__(
        self, params: ndarray, x: Optional[ndarray] = None
    ) -> Tuple[Circuit, ndarray, ndarray, dict]:

        if (
            any(isinstance(circ, QuantumEncoding) for circ in self.circuit_structure)
            and x is None
        ):
            raise ValueError(
                "x cannot be None when encoding layers are present in the circuit structure."
            )

        # the complete circuit
        circuit = None
        # the jacobian for each sub-circuit
        jacobians = []
        jacobians_wrt_inputs = []
        input_to_gate_map = {}
        param_to_gate_map = {}

        index = 0
        for circ in self.circuit_structure:
            if isinstance(circ, QuantumEncoding):
                # encoders do not have parameters
                nparams = 0
                jacobian, input_map, circ = self._build_from_encoding(
                    circ, x, trace=True
                )
                # update the input_map to the index of the global circuit
                input_map = {
                    inp: tuple(i + index for i in indices)
                    for inp, indices in input_map.items()
                }
                # update the global map
                for inp, indices in input_map.items():
                    if inp in input_to_gate_map:
                        input_to_gate_map[inp] += indices
                    else:
                        input_to_gate_map[inp] = indices
                jacobians_wrt_inputs.append(jacobian)
            elif isinstance(circ, Circuit):
                nparams = len(circ.get_parameters())
                jacobian, par_map, circ = self._build_from_circuit(
                    circ, params[index : index + nparams], trace=True
                )
                jacobians.append(jacobian)
            else:
                param_dict = signature(circ).parameters
                nparams = len(param_dict)
                jacobian, par_map, circ = self._build_from_callable(
                    circ, params[index : index + nparams], trace=True
                )
                jacobians.append(jacobian)
            index += nparams
            if circuit is None:
                circuit = circ
            else:
                circuit += circ

        total_dim = tuple(sum(np.array(j.shape) for j in jacobians))
        # build the global jacobian
        J = self.zeros(
            total_dim, dtype=self._get_dtype(params), device=self._get_device(params)
        )
        position = np.array([0, 0])
        # insert each sub-jacobian in the total one
        for j in jacobians:
            shape = np.array(j.shape)
            interval = tuple(zip(position, shape + position))
            J[interval[0][0] : interval[0][1], interval[1][0] : interval[1][1]] = j
            position += shape

        return circuit, self.engine.vstack(jacobians_wrt_inputs), J, input_to_gate_map

    def build_circuit(self, params: ndarray, x: Optional[ndarray] = None) -> Circuit:
        circuit = None

        index = 0
        for circ in self.circuit_structure:
            if isinstance(circ, QuantumEncoding):
                nparams = 0
                circ = self._build_from_encoding(circ, x, trace=False)
            elif isinstance(circ, Circuit):
                nparams = len(circ.get_parameters())
                circ = self._build_from_circuit(
                    circ, params[index : index + nparams], trace=False
                )
            else:
                param_dict = signature(circ).parameters
                nparams = len(param_dict)
                circ = self._build_from_callable(
                    circ, params[index : index + nparams], trace=False
                )
            index += nparams
            if circuit is None:
                circuit = circ
            else:
                circuit += circ
        return circuit
