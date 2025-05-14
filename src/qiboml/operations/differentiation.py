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
        # construct circuit using the full circuit helper
        circuit = circuit_from_structure(
            circuit_structure=circuit_structure,
            x=x,
        )
        gradient = []
        if wrt_inputs:
            # compute first gradient part, wrt data
            gradient.append(
                backend.np.reshape(
                    self.gradient_wrt_inputs(
                        x=x,
                        parameters=parameters,
                        circuit_structure=circuit_structure,
                        decoding=decoding,
                        backend=backend,
                    ),
                    (decoding.output_shape[-1], x.shape[-1]),
                )
            )
        else:
            gradient.append(
                backend.np.zeros(
                    (decoding.output_shape[-1], x.shape[-1]), dtype=x.dtype
                )
            )

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

    def gradient_wrt_inputs(self, x, parameters, circuit_structure, decoding, backend):
        """Compute the gradient of the given model w.r.t inputs."""

        gradient = backend.np.zeros(
            (decoding.output_shape[-1], x.shape[-1]), dtype=x.dtype
        )

        forwards, backwards, eigenvals = self.dx_circuits(
            x=x,
            parameters=parameters,
            circuit_structure=circuit_structure,
            backend=backend,
        )

        for xi in range(len(gradient)):
            for fwd, bwd, eig in zip(forwards[xi], backwards[xi], eigenvals[xi]):
                gradient[xi] += float(decoding(fwd) - decoding(bwd)) * eig

    def dx_circuits(
        self,
        x: ndarray,
        parameters: ndarray,
        circuit_structure: List[Union[Circuit, QuantumEncoding]],
        backend: Backend,
    ):
        """
        Collect all the forward and backward circuits required to compute the
        gradient w.r.t. the input data.
        """
        n_inputs = x.shape[-1]
        forwards, backwards, eigenvals = [] * n_inputs, [] * n_inputs, [] * n_inputs
        for xi in range(n_inputs):
            forwards[xi], backwards[xi], eigenvals[xi] = (
                self.dxi_circuits_from_encodings(
                    xi=xi,
                    x=x,
                    parameters=parameters,
                    circuit_structure=circuit_structure,
                    backend=backend,
                )
            )
        return forwards, backwards, eigenvals

    def dxi_circuits_from_encodings(
        self,
        xi,
        x,
        parameters: ndarray,
        circuit_structure: List[Union[Circuit, QuantumEncoding]],
        backend: Backend,
    ):
        """
        Construct the forward and backward circuits required to compute the
        derivative of the expectation value w.r.t the i-th component of the input
        data `x` considering all the encoding layers.
        """
        forwards, backwards, eigenvals = [], [], []
        for circ_index, circ in enumerate(circuit_structure):
            # When we encounter an encoding layer
            if isinstance(circ, QuantumEncoding):
                encoding_layer = circ(x)
                affected_gates = encoding_layer._data_to_gate[str(xi)]
                # Retrieve - one by one - gates affected by xi
                for affected_index in affected_gates:
                    gate = encoding_layer.queue[affected_index]

                    if len(gate.parameters) != 1:
                        raise_error(  # pragma: no cover (we are covering an equivalent error in the encoding itself)
                            NotImplementedError,
                            "For now, shift rules are supported for 1-parameter gates only.",
                        )
                    else:
                        # Calculate the shift amount based on the generator eigenvalue.
                        # TODO: this is true only for gates of the type introduced in https://arxiv.org/abs/1811.11184
                        gen_eigenval = gate.generator_eigenvalue()
                        shift = np.pi / (4 * gen_eigenval)
                        # Build the forward-shifted circuit.
                        forwards.append(
                            self.shifted_circuit(
                                x=x,
                                parameters=parameters,
                                angle_index=xi,
                                circuit_index=circ_index,
                                affected_index=affected_index,
                                shift=shift,
                                circuit_structure=circuit_structure,
                                backend=backend,
                            )
                        )
                        # Build the backward-shifted circuit.
                        backwards.append(
                            self.shifted_circuit(
                                x=x,
                                parameters=parameters,
                                angle_index=xi,
                                circuit_index=circ_index,
                                affected_index=affected_index,
                                shift=-shift,
                                circuit_structure=circuit_structure,
                                backend=backend,
                            )
                        )
                        eigenvals.append(gen_eigenval)
        return forwards, backwards, eigenvals

    def shifted_circuit(
        self,
        x: ndarray,
        parameters: ndarray,
        angle_index: int,
        affected_index: Optional[int],
        circuit_index: int,
        shift: float,
        circuit_structure: List[Union[QuantumEncoding, Circuit]],
        backend: Backend,
    ):
        """
        Helper function to reconstruct a circuit by shifting only one of the
        angles.
        """

        # Shift data or parameters depending on the task
        if isinstance(circuit_structure[circuit_index], QuantumEncoding):
            tmp_circ = self.shift_encoding_gate(
                encoding_layer=circuit_structure[circuit_index](x),
                affected_index=affected_index,
                shift=shift,
            )
        else:
            tmp_array = backend.cast(parameters, copy=True)
            tmp_array = self.shift_parameter(tmp_array, angle_index, shift, backend)
            tmp_circ = deepcopy(circuit_structure[circuit_index])
            tmp_circ.set_parameters(tmp_array)

        nqubits = circuit_structure[0].nqubits
        circuit = Circuit(nqubits)
        # Build circuit before the target layer
        if not circuit_index == 0:
            circuit += circuit_from_structure(
                circuit_structure=circuit_structure[:circuit_index],
                x=x,
            )
        # Build target layer
        circuit += tmp_circ
        # Build circuit after target layer
        if not circuit_index == len(circuit_index) - 1:
            circuit += circuit_from_structure(
                circuit_structure=circuit_structure[circuit_index + 1 :],
                x=x,
            )
        return circuit

    @staticmethod
    def shift_encoding_gate(
        encoding_layer,
        affected_index,
        shift,
    ):
        """
        Modify a given encoding layer shifting the angle of the `affected_index`
        gate affected by `xi`.
        """
        tmp_circ = deepcopy(encoding_layer)
        affected_gate = tmp_circ.queue[affected_index]
        # TODO: adapt this to other backends (now working with Numpy-like)
        affected_gate.parameters += shift
        return tmp_circ

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


class _StaticArgWrapper:
    """
    A thin wrapper to make non-hashable objects hashable by using their id.
    """

    def __init__(self, obj):
        self.obj = obj

    def __hash__(self):
        return id(self.obj)

    def __eq__(self, other):
        return id(self.obj) == id(other.obj)


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
        x = backend.to_numpy(x)
        x = self._jax.cast(x, self._jax.precision)

        circuit_structure_static = _StaticArgWrapper(circuit_structure)
        decoding_static = _StaticArgWrapper(decoding)

        if self._argnums is None:
            self._argnums = (0, 3)
            self._jacobian = partial(jax.jit, static_argnums=(1, 2))(
                jax.jacfwd(self._run, (0, 3))
            )

        params = np.array(list(parameters))  # shape: (n_params,)
        params = self._jax.cast(params, params.dtype)
        decoding.set_backend(self._jax)

        grad_inputs, grad_params = self._jacobian(
            x, circuit_structure_static, decoding_static, params
        )

        if not wrt_inputs:
            grad_inputs = self._jax.numpy.zeros(
                (decoding.output_shape[-1], x.shape[-1])
            )

        if grad_params.ndim == 3 and grad_params.shape[1] == 1:
            grad_params = self._jax.numpy.squeeze(grad_params, axis=1)
        num_params = int(grad_params.shape[1])
        split_param_grads = list(self._jax.numpy.split(grad_params, num_params, axis=1))
        decoding.set_backend(backend)

        return [
            backend.cast(self._jax.to_numpy(grad).tolist(), backend.precision)
            for grad in ([grad_inputs] + split_param_grads)
        ]

    @staticmethod
    @partial(jax.jit, static_argnums=(1, 2))
    def _run(x, circuit_structure_static, decoding_static, parameters):
        circuit_structure = circuit_structure_static.obj
        decoding = decoding_static.obj
        circ = circuit_from_structure(
            circuit_structure=circuit_structure,
            x=x,
        )
        circ.set_parameters(parameters)
        return decoding(circ)
