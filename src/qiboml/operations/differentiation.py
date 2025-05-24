from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from typing import Callable, List, Union

import jax
import numpy as np
from qibo import Circuit
from qibo.backends import Backend
from qibo.config import raise_error

from qiboml import ndarray
from qiboml.backends.jax import JaxBackend
from qiboml.interfaces.utils import circuit_from_structure
from qiboml.models.decoding import QuantumDecoding
from qiboml.models.encoding import QuantumEncoding


@dataclass
class Differentiation(ABC):
    """
    Abstract differentiator object.
    """

    @abstractmethod
    def evaluate(
        self,
        x: ndarray,
        circuit_structure: List[Union[Circuit, QuantumEncoding]],
        decoding: QuantumDecoding,
        backend: Backend,
        *parameters: list[ndarray],
        wrt_inputs: bool = False,
    ):  # pragma: no cover
        """
        Evaluate the gradient of a quantum model w.r.t inputs and parameters.
        """
        pass


class PSR(Differentiation):
    """
    The Parameter Shift Rule differentiator. Especially useful for non analytical
    derivative calculation which, thus, makes it hardware compatible.
    """

    def __init__(self):
        self.data_map = {}
        self._data_map_cached = False

    def update_data_map(self, circuit_structure, backend, x):
        """
        Update the map object which is storing the encoding rules of the whole
        quantum circuit. Namely, it can be used to retrieve which gates are
        affected by the input data component `x[i]` accessing `psr.data_map['i']`.
        """
        # Update the data map if the given shape has never been used
        if not self._data_map_cached:
            self.data_map = self._build_data_map(
                circuit_structure, backend, x.shape[-1]
            )
            self._data_map_cached = True

    def _build_data_map(self, circuit_structure, backend, shape):
        """
        Helper method to build and cache the data map structure only the first
        time the `psr.evaluate` method is called.
        """
        # Construct the map with placeholders
        data_map = {str(i): [] for i in range(shape)}

        # Construct a dummy data of the given shape
        dummy_x = backend.np.zeros(shape)

        gate_index = 0
        for circ in circuit_structure:
            if isinstance(circ, QuantumEncoding):
                for xi in range(shape):
                    # Update the map adding the indices affected by xi and placed
                    # in the proper position of the whole circuit queue
                    data_map[str(xi)].extend(
                        [i + gate_index for i in circ._data_to_gate[str(xi)]]
                    )
                gate_index += len(circ(dummy_x).queue)
            else:
                gate_index += len(circ.queue)

        return data_map

    def evaluate(
        self,
        x: ndarray,
        circuit_structure: List[Union[Circuit, QuantumEncoding]],
        decoding: QuantumDecoding,
        backend: Backend,
        *parameters: List[ndarray],
        wrt_inputs: bool = False,
    ):
        """
        Evaluate the gradient of a quantum model w.r.t inputs and parameters,
        respectively represented by `x` and `parameters`.
        Args:
            x (ndarray): the input data.
            circuit_structure (List[Union[Circuit, QuantumEncoding]]): structure
                of the circuit. It can be composed of `QuantumEncoding`s and
                Qibo's circuits.
            decoding (QuantumDecoding): the decoding layer.
            backend (Backend): the backend to execute the circuit with.
            parameters (List[ndarray]): the parameters at which to evaluate the model, and thus the derivative.
            wrt_inputs (bool): whether to calculate the derivate with respect to inputs or not, by default ``False``.
        Returns:
            (list[ndarray]): the calculated gradients.
        """
        if decoding.output_shape != (1, 1):
            raise_error(
                NotImplementedError,
                "Parameter Shift Rule only supports expectation value decoding.",
            )

        # Construct circuit using the full circuit helper
        circuit = circuit_from_structure(
            circuit_structure=circuit_structure,
            x=x,
        )

        # Inject parameters into the circuit
        circuit.set_parameters(parameters)

        gradient = []
        if wrt_inputs:
            # Compute once the data map to know how input x affects the
            # circuit gates
            self.update_data_map(
                circuit_structure=circuit_structure,
                backend=backend,
                x=x,
            )
            # Compute first gradient part, wrt input data
            gradient.append(
                backend.np.reshape(
                    self.gradient_wrt_inputs(
                        x=x,
                        circuit=circuit,
                        decoding=decoding,
                        backend=backend,
                    ),
                    (decoding.output_shape[-1], x.shape[-1]),
                )
            )
        elif x is not None:
            gradient.append(
                backend.np.zeros(
                    (decoding.output_shape[-1], x.shape[-1]), dtype=x.dtype
                )
            )
        else:
            gradient.append(backend.cast([]))

        # compute second gradient part, wrt parameters
        for i in range(len(parameters)):
            gradient.append(
                self.one_parameter_shift(
                    circuit=circuit,
                    decoding=decoding,
                    parameters=parameters,
                    parameter_index=i,
                    backend=backend,
                )
            )
        return gradient

    def one_parameter_shift(
        self, circuit, decoding, parameters, parameter_index, backend
    ):
        """Compute one derivative of the decoding strategy w.r.t. a target parameter."""
        gate = circuit.associate_gates_with_parameters()[parameter_index]
        generator_eigenval = gate.generator_eigenvalue()
        s = np.pi / (4 * generator_eigenval)

        tmp_params = backend.cast(parameters, copy=True, dtype=parameters[0].dtype)
        tmp_params = self.shift_parameter(tmp_params, parameter_index, s, backend)

        circuit.set_parameters(tmp_params)
        forward = decoding(circuit)

        tmp_params = self.shift_parameter(tmp_params, parameter_index, -2 * s, backend)

        circuit.set_parameters(tmp_params)
        backward = decoding(circuit)
        return generator_eigenval * (forward - backward)

    def gradient_wrt_inputs(self, x, circuit, decoding, backend):
        """Compute the gradient of the given model w.r.t inputs."""

        gradient = backend.np.zeros(x.shape[-1], dtype=x.dtype)

        forwards, backwards, eigenvals = self.dx_circuits(
            x=x,
            circuit=circuit,
        )

        for xi in range(len(gradient)):
            for fwd, bwd, eig in zip(forwards[xi], backwards[xi], eigenvals[xi]):
                gradient[xi] += float(decoding(fwd) - decoding(bwd)) * eig

        return gradient

    def dx_circuits(
        self,
        x: ndarray,
        circuit: Circuit,
    ):
        """
        Collect all the forward and backward circuits required to compute the
        gradient w.r.t. the input data.
        """
        # TODO: consider to flatten the data at the beginning of the evaluate
        # process
        n_inputs = x.shape[-1]
        forwards, backwards, eigenvals = (
            [None] * n_inputs,
            [None] * n_inputs,
            [None] * n_inputs,
        )
        for xi in range(n_inputs):
            forwards[xi], backwards[xi], eigenvals[xi] = self.dxi_circuits(
                xi=xi,
                circuit=circuit,
            )
        return forwards, backwards, eigenvals

    def dxi_circuits(
        self,
        xi,
        circuit: Circuit,
    ):
        """
        Construct the forward and backward circuits required to compute the
        derivative of the expectation value w.r.t the i-th component of the input
        data `x` considering all the encoding layers.
        """
        forwards, backwards, eigenvals = [], [], []

        for ig in self.data_map[str(xi)]:
            if len(circuit.queue[ig].parameters) != 1:
                raise_error(  # pragma: no cover (we are covering an equivalent error in the encoding itself)
                    NotImplementedError,
                    "For now, shift rules are supported for 1-parameter gates only.",
                )
            eigenval = circuit.queue[ig].generator_eigenvalue()
            shift = np.pi / (4 * eigenval)

            forward = deepcopy(circuit)
            # TODO: we deal with tuple so we have to fix this
            # TODO: this is only valid when the gate has 1 parameter for now
            original_parameter = deepcopy(circuit.queue[ig].parameters[0])
            forward.queue[ig].parameters = original_parameter + shift
            backward = deepcopy(circuit)
            backward.queue[ig].parameters = original_parameter - shift

            forwards.append(forward)
            backwards.append(backward)
            eigenvals.append(eigenval)

        return forwards, backwards, eigenvals

    @staticmethod
    def shift_parameter(parameters, i, epsilon, backend):
        if backend.platform == "tensorflow":
            return backend.tf.stack(
                [parameters[j] + int(i == j) * epsilon for j in range(len(parameters))]
            )
        elif backend.platform == "jax":
            parameters = parameters.at[i].set(parameters[i] + epsilon)
        else:
            parameters[i] = parameters[i] + epsilon
        return parameters


class Jax(Differentiation):
    """
    The Jax differentiator object. Particularly useful for enabling gradient calculation in
    those backends that do not provide it. Note, however, that for this reason the circuit is
    executed with the JaxBackend whenever a derivative is needed.
    """

    def __init__(self):
        self._jax: Backend = JaxBackend()
        self._argnums: tuple[int] = None
        self._jacobian: Callable = lambda *args, **kwargs: None
        self._jacobian_without_inputs: Callable = lambda *args, **kwargs: None

    def evaluate(
        self,
        x: ndarray,
        circuit_structure: List[Union[Circuit, QuantumEncoding]],
        decoding: QuantumDecoding,
        backend: Backend,
        *parameters: list[ndarray],
        wrt_inputs: bool = False,
    ):
        """
        Evaluate the gradient of a quantum model w.r.t inputs and parameters,
        respectively represented by `x` and `parameters`.
        Args:
            x (ndarray): the input data.
            circuit_structure (List[Union[Circuit, QuantumEncoding]]): structure
                of the circuit. It can be composed of `QuantumEncoding`s and
                Qibo's circuits.
            decoding (QuantumDecoding): the decoding layer.
            backend (Backend): the backend to execute the circuit with.
            parameters (list[ndarray]): the parameters at which to evaluate the model, and thus the derivative.
            wrt_inputs (bool): whether to calculate the derivate with respect to inputs or not, by default ``False``.
        Returns:
            (list[ndarray]): the calculated gradients.
        """
        if x is not None:
            x = backend.to_numpy(x)
            x = self._jax.cast(x, self._jax.precision)

        circuit_structure = tuple(circuit_structure)

        if self._argnums is None:
            self._argnums = tuple(range(3, len(parameters) + 3))
            setattr(
                self,
                "_jacobian",
                partial(jax.jit, static_argnums=(1, 2))(
                    jax.jacfwd(self._run, (0,) + self._argnums),
                ),
            )
            setattr(
                self,
                "_jacobian_without_inputs",
                partial(jax.jit, static_argnums=(1, 2))(
                    jax.jacfwd(self._run, self._argnums),
                ),
            )

        parameters = np.array(backend.to_numpy(list(parameters)))
        parameters = self._jax.cast(parameters, parameters.dtype)

        decoding.set_backend(self._jax)

        if wrt_inputs:
            gradients = self._jacobian(  # pylint: disable=no-member
                x, circuit_structure, decoding, *parameters
            )
        else:
            shape = 0 if x is None else (decoding.output_shape[-1], x.shape[-1])
            gradients = (
                self._jax.numpy.zeros(shape),
                self._jacobian_without_inputs(  # pylint: disable=no-member
                    x, circuit_structure, decoding, *parameters
                ),
            )
        decoding.set_backend(backend)
        return [
            backend.cast(self._jax.to_numpy(grad).tolist(), backend.precision)
            for grad in gradients
        ]

    @staticmethod
    @partial(jax.jit, static_argnums=(1, 2))
    def _run(x, circuit_structure, decoding, *parameters):
        circ = circuit_from_structure(
            circuit_structure=circuit_structure,
            x=x,
        )
        circ.set_parameters(parameters)
        return decoding(circ)
