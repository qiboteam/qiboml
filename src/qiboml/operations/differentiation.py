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
        nlayers: int,
        *parameters: list[ndarray],
        wrt_inputs: bool = False,
    ):
        """
        Evaluate the gradient of a quantum model w.r.t inputs and parameters.
        """
        pass

    @staticmethod
    def full_circuit(
        x: ndarray,
        nqubits: int,
        circuit_structure: List[Union[Circuit, QuantumEncoding]],
    ):
        """
        Helper method to reconstruct the full circuit covering the reuploading strategy.
        """
        circuit = Circuit(nqubits)
        for circ in circuit_structure:
            if isinstance(circ, QuantumEncoding):
                circuit += circ(x)
            else:
                circuit += circ
        return circuit


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
        *parameters: list[ndarray],
        wrt_inputs: bool = False,
    ):
        if decoding.output_shape != (1, 1):
            raise_error(
                NotImplementedError,
                "Parameter Shift Rule only supports expectation value decoding.",
            )
        # construct circuit using the full circuit helper
        circuit = Differentiation.full_circuit(
            x, circuit_structure[0].nqubits, circuit_structure
        )
        gradient = []
        if wrt_inputs:
            # compute first gradient part, wrt data
            gradient.append(
                backend.np.reshape(
                    backend.np.hstack(
                        self.gradient_wrt_inputs(
                            x=x,
                            parameters=parameters,
                            circuit_structure=circuit_structure,
                            decoding=decoding,
                            backend=backend,
                        )
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

        # The gradient w.r.t. x has to be the same dimension of x
        gradient = backend.np.zeros(len(x))

        # Scanning the circuit structure to search encodings
        for i_circ, circ in enumerate(circuit_structure):
            if isinstance(circ, QuantumEncoding):
                # For each encoding, we need to track input vals locations
                gradient += self.data_gradient_from_encoding(
                    x=x,
                    parameters=parameters,
                    circ_index=i_circ,
                    circuit_structure=circuit_structure,
                    decoding=decoding,
                    backend=backend,
                )

        return gradient

    def data_gradient_from_encoding(
        self,
        x: ndarray,
        parameters: ndarray,
        circ_index: int,
        circuit_structure: List[Union[Circuit, QuantumEncoding]],
        decoding: QuantumDecoding,
        backend: Backend,
    ):
        """
        Compute the contribution to the gradient w.r.t inputs due to a specific
        encoding layer in the circuit structure.
        """
        gradient = backend.np.zeros(len(x))

        for i_x in range(len(x)):
            # Collecting the indices of the gates, in the queue of encoding
            # which are affected by the i-th component of x
            affected_gates_indices = circuit_structure[circ_index]._data_to_gate[
                str(i_x)
            ]
            for gate in [
                circuit_structure[circ_index](x).queue[i]
                for i in affected_gates_indices
            ]:
                if len(gate.parameters) != 1:
                    raise_error(
                        NotImplementedError,
                        "For now, shift rules are supported for 1-parameters gates only.",
                    )
                else:
                    shift = np.pi / (4 * gate.generator_eigenvalue())
                    # Forward circuit
                    forward = self.shifted_circuit(
                        x=x,
                        parameters=parameters,
                        angle_index=i_x,
                        circuit_index=circ_index,
                        shift=shift,
                        circuit_structure=circuit_structure,
                        backend=backend,
                    )
                    # Backward circuit
                    backward = self.shifted_circuit(
                        x=x,
                        parameters=parameters,
                        angle_index=i_x,
                        circuit_index=circ_index,
                        shift=-shift,
                        circuit_structure=circuit_structure,
                        backend=backend,
                    )
                    gradient[i_x] += (
                        decoding(forward) - decoding(backward)
                    ) * gate.generator_eigenvalue()
        return gradient

    def shifted_circuit(
        self,
        x: ndarray,
        parameters: ndarray,
        angle_index: int,
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
            tmp_array = backend.cast(x, copy=True)
            tmp_array = self.shift_parameter(tmp_array, angle_index, shift, backend)
            tmp_circ = deepcopy(circuit_structure[circuit_index](tmp_array))
        else:
            tmp_array = backend.cast(parameters, copy=True)
            tmp_array = self.shift_parameter(tmp_array, angle_index, shift, backend)
            tmp_circ = deepcopy(circuit_structure[circuit_index])
            tmp_circ.set_parameters(tmp_array)

        nqubits = circuit_structure[0].nqubits
        circuit = Circuit(nqubits)
        circuit += Differentiation.full_circuit(
            x, nqubits=nqubits, circuit_structure=circuit_structure[:circuit_index]
        )
        circuit += tmp_circ
        circuit += Differentiation.full_circuit(
            x, nqubits=nqubits, circuit_structure=circuit_structure[circuit_index + 1 :]
        )
        return circuit

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
        x = backend.to_numpy(x)
        x = self._jax.cast(x, self._jax.precision)
        # Wrap the static arguments (non-hashable objects) so they can be used as static args.
        circuit_structure_static = _StaticArgWrapper(circuit_structure)
        decoding_static = _StaticArgWrapper(decoding)
        if self._argnums is None:
            self._argnums = tuple(range(3, len(parameters) + 3))
            setattr(
                self,
                "_jacobian",
                partial(jax.jit, static_argnums=(1, 2))(
                    jax.jacfwd(self._run, (0,) + self._argnums)
                ),
            )
            setattr(
                self,
                "_jacobian_without_inputs",
                partial(jax.jit, static_argnums=(1, 2))(
                    jax.jacfwd(self._run, self._argnums)
                ),
            )
        parameters = backend.to_numpy(list(parameters))
        parameters = self._jax.cast(parameters, parameters.dtype)
        decoding.set_backend(self._jax)
        if wrt_inputs:
            gradients = self._jacobian(
                x,
                circuit_structure_static,
                decoding_static,
                *parameters,
            )
        else:
            gradients = (
                self._jax.numpy.zeros((decoding.output_shape[-1], x.shape[-1])),
                self._jacobian_without_inputs(
                    x,
                    circuit_structure_static,
                    decoding_static,
                    *parameters,
                ),
            )
        decoding.set_backend(backend)
        return [
            backend.cast(self._jax.to_numpy(grad).tolist(), backend.precision)
            for grad in gradients
        ]

    @staticmethod
    @partial(jax.jit, static_argnums=(1, 2))
    def _run(x, circuit_structure_static, decoding_static, *parameters):
        # Unwrap the static objects
        circuit_structure = circuit_structure_static.obj
        decoding = decoding_static.obj

        # Build the full circuit using the shared helper from Differentiation.
        circ = Differentiation.full_circuit(
            x, circuit_structure[0].nqubits, circuit_structure
        )
        circ.set_parameters(parameters)
        return decoding(circ)
