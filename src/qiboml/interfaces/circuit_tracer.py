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
    """Wrapper to the circuit structure that takes care of tracing the circuit construction.
    In particular, it computes both the Jacobian wrt the parameters and the inputs of the operations
    that construct the circuits.
    """

    circuit_structure: List[Union[Circuit, QuantumEncoding, Callable]]
    derivation_mode: str = "forward"

    @property
    @abstractmethod
    def engine(self):  # pragma: no cover
        """The calculation engine used."""
        pass

    @staticmethod
    @abstractmethod
    def jacfwd(f: Callable, argnums: Union[int, Tuple[int]]):  # pragma: no cover
        """The jacobian functional in forward derivation mode."""
        pass

    @staticmethod
    @abstractmethod
    def jacrev(f: Callable, argnums: Union[int, Tuple[int]]):  # pragma: no cover
        """The jacobian functional in reverse derivation mode."""
        pass

    def _compute_jacobian_functional(
        self, circuit: Union[QuantumEncoding, Callable]
    ) -> Callable:
        """Compute the jacobian functional for the input function.

        Args:
            circuit (Callable): the input functions that build the circuit.

        Returns:
            (Callable) the jacobian functional
        """
        _tmp_circuit = circuit

        def build(x):
            is_encoding = isinstance(_tmp_circuit, QuantumEncoding)
            if is_encoding:
                circuit = _tmp_circuit(x)
            else:
                # this is needed for symbolic execution with tf
                x = [x[i] for i in range(len(x))]
                circuit = _tmp_circuit(*x)
            out = self.engine.stack(
                [
                    p
                    for pars in circuit.get_parameters(
                        include_not_trainable=is_encoding
                    )
                    for p in pars
                ]
            )
            return self.engine.reshape(out, (-1, 1))

        jac = self.jacfwd if self.derivation_mode == "forward" else self.jacrev
        return jac(build, argnums=0)

    @cached_property
    def jacobian_functionals(
        self,
    ) -> dict[int, Callable]:
        """The dictionary containing the jacobian functionals of each element composing
        circuit structure.
        """
        jacobians = {}
        for circ in self.circuit_structure:
            if isinstance(circ, Circuit):
                jacobians[id(circ)] = self.identity
            else:
                jacobians[id(circ)] = self._compute_jacobian_functional(circ)
        return jacobians

    def nonzero(self, array: ndarray) -> ndarray:
        """The numpy-like np.nonzero function of the current engine."""
        return self.engine.nonzero(array)

    def _build_parameters_map(self, jacobian: ndarray) -> dict[int, Tuple[int]]:
        """Construct the mapping between independent and dependent parameters.
        In detail, the index of each independent parameter is mapped to the indices
        of all the dependent parameters that originated from it. This is particularly
        useful for the jacobian of encoders, as it provides a map from the inputs to
        the actual gates where they are encoded in the circuit.
        """
        par_map = {}
        for i, row in enumerate(jacobian):
            for j in self.nonzero(row):
                j = int(j)
                if j in par_map:
                    par_map[j] += (i,)
                else:
                    par_map[j] = (i,)
        return par_map

    def trace(
        self, f: Union[Callable, QuantumEncoding], params: ndarray
    ) -> Tuple[ndarray, dict[int, Tuple[int]]]:
        """Trace the construction of a circuit through the function `f` with given
        parameters `params`.

        Args:
            f (Callable): the function that builds the circuit, either a custom user-defined
        function or an encoder.
            params (ndarray): the parameters assigned to the built circuit.

        Returns:
            (Tuple(ndarray, dict)) the computed jacobian and the mapping between the independent
        and dependent parameters of `f`.
        """
        # we always assume the input is a 1-dim array, even for encodings
        # thus the jacobian is always a matrix
        jac = self.engine.reshape(
            self.jacobian_functionals[id(f)](params), (-1, params.shape[0])
        )
        par_map = self._build_parameters_map(jac)
        return jac, par_map

    @cached_property
    def is_encoding_differentiable(self) -> bool:
        """Check if all the encoders in the circuit structure are differentiable."""
        diff_encodings = [
            circ.differentiable
            for circ in self.circuit_structure
            if isinstance(circ, QuantumEncoding)
        ]
        if len(diff_encodings) == 0:
            return False
        # all the encodings must be differentiable
        # open to debate, not the only possible choice
        return all(diff_encodings)

    @abstractmethod
    def requires_gradient(self, x: ndarray) -> bool:  # pragma: no cover
        """Check whether the input array needs gradients to be calculated for it."""
        pass

    def _build_from_encoding(
        self, encoding: QuantumEncoding, x: ndarray, trace: bool = True
    ) -> Circuit:
        """Build the circuit starting from an encoder.

        Args:
            encoding (QuantumEncoding): the encoder.
            x (ndarray): the input data.
            trace (bool): whether to trace the construction and thus also
            calculate the jacobian. Defaults to ``True``.

        Returns:
            (Circuit | Tuple(ndarray, dict, Circuit)) the built circuit or the
        tuple: jacobian wrt inputs, input to gate map and circuit.
        """
        circuit = encoding(x)
        if trace:
            if self.is_encoding_differentiable and self.requires_gradient(x):
                if len(x.shape) > 1:
                    x = x[0]
                return *self.trace(encoding, x), circuit
            return None, None, circuit
        return circuit

    def _build_from_circuit(
        self, circuit: Circuit, params: ndarray, trace: bool = True
    ) -> Circuit:
        """Build the circuit starting from a circuit. In practice the given
        parameters `params` are set into the circuit only.

        Args:
            circuit (Circuit): the circuit.
            params (ndarray): the parameters to set.
            trace (bool): whether to trace the construction and thus also
            calculate the jacobian. Defaults to ``True``.

        Returns:
            (Circuit | Tuple(ndarray, dict, Circuit)) the built circuit or the
        tuple: jacobian wrt parameters, input to gate map and circuit.
        """
        circuit.set_parameters(params)
        if trace:
            jacobian = self.identity(
                params.shape[0],
                dtype=self._get_dtype(params),
                device=self._get_device(params),
            )
            # all the circuit parameters are considered independent
            par_map = {i: (i,) for i in range(params.shape[0])}
            return jacobian, par_map, circuit
        return circuit

    def _build_from_callable(
        self, f: Callable, params: ndarray, trace: bool = True
    ) -> Circuit:
        """Build the circuit starting from a custom callable.

        Args:
            f (Callable): the custom callable which builds the circuit.
            params (ndarray): the parameters used for construction.
            trace (bool): whether to trace the construction and thus also
            calculate the jacobian. Defaults to ``True``.

        Returns:
            (Circuit | Tuple(ndarray, dict, Circuit)) the built circuit or the
        tuple: jacobian wrt parameters, input to gate map and circuit.
        """
        circuit = f(*params)
        if trace:
            return *self.trace(f, params), circuit
        return circuit

    @staticmethod
    def _get_device(array: ndarray):
        """Extract the device of the input array."""
        return array.device

    @staticmethod
    def _get_dtype(array: ndarray):
        """Extract the dtype of the input array."""
        return array.dtype

    @abstractmethod
    def identity(self, dim: int, dtype, device) -> ndarray:  # pragma: no cover
        """The numpy-like np.eye function of the current engine."""
        pass

    @abstractmethod
    def zeros(
        self, shape: Union[int, Tuple[int]], dtype, device
    ) -> ndarray:  # pragma: no cover
        """The numpy-like np.zeros function of the current engine."""
        pass

    def fill_jacobian(
        self,
        jacobian: ndarray,
        row_span: Tuple[int, int],
        col_span: Tuple[int, int],
        values: ndarray,
    ) -> ndarray:
        """Fill the input jacobian in the span defined by (row_span, col_span) with
        the given values. This is mostly here to be overwritten by engines that do not
        allow for direct item assignment.

        Args:
            jacobian (ndarray): the jacobian to fill.
            row_span (Tuple[int, int]): the span of row indices.
            col_span (Tuple[int, int]): the span of column indices.
            values (ndarray): the values to insert.

        Returns:
            (ndarray) the filled jacobian.
        """
        jacobian[row_span[0] : row_span[1], col_span[0] : col_span[1]] = values
        return jacobian

    def __call__(
        self, params: ndarray, x: Optional[ndarray] = None
    ) -> Tuple[Circuit, Optional[ndarray], ndarray, Optional[dict]]:
        """Construct the circuit defined by the internal ``circuit_structure`` and compute
        the jacobian wrt to the parameters `params` and inputs `x` of the construction process.

        Args:
            params (ndarray): the parameters to construct the circuit with.
            x (ndarray, optional): the inputs to construct the circuit with.

        Returns:
            (Tuple(Circuit, ndarray, ndarray, dict)) the constructed circuit, jacobian wrt inputs,
        jacobian wrt parameters and mapping of inputs ``x`` to the corresponding gate in the built circuit.
        """
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

        index = 0
        angle_offset = 0
        for circ in self.circuit_structure:
            if isinstance(circ, QuantumEncoding):
                # encoders do not have parameters
                nparams = 0
                jacobian, input_map, circ = self._build_from_encoding(
                    circ, x, trace=True
                )
                if input_map is not None:
                    # update the input_map to the index of the global circuit
                    input_map = {
                        # inp: tuple(i + index for i in indices)
                        inp: tuple(i + angle_offset for i in indices)
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
                jacobian, _, circ = self._build_from_circuit(
                    circ, params[index : index + nparams], trace=True
                )
                jacobians.append(jacobian)
            else:
                param_dict = signature(circ).parameters
                nparams = len(param_dict)
                jacobian, _, circ = self._build_from_callable(
                    circ, params[index : index + nparams], trace=True
                )
                jacobians.append(jacobian)
            index += nparams
            if circuit is None:
                circuit = circ
            else:
                circuit += circ

            angle_offset += sum(
                len(pars) for pars in circ.get_parameters(include_not_trainable=True)
            )

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
            J = self.fill_jacobian(J, interval[0], interval[1], j)
            # direct assignment works with torch/numpy only
            # J[interval[0][0] : interval[0][1], interval[1][0] : interval[1][1]] = j
            position += shape

        jacobians_wrt_inputs = (
            None
            if len(jacobians_wrt_inputs) == 0 or jacobians_wrt_inputs[0] is None
            else self.engine.vstack(jacobians_wrt_inputs)
        )
        if len(input_to_gate_map) == 0:
            input_to_gate_map = None

        return circuit, jacobians_wrt_inputs, J, input_to_gate_map

    def build_circuit(self, params: ndarray, x: Optional[ndarray] = None) -> Circuit:
        """Construct the circuit defined by the internal ``circuit_structure`` without
        worrying about tracing the operations.

        Args:
            params (ndarray): the parameters to construct the circuit with.
            x (ndarray, optional): the inputs to construct the circuit with.

        Returns:
            (Circuit) the constructed circuit.
        """
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
