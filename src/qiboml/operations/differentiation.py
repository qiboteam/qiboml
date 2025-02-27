from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial
from typing import List, Union

import jax
import numpy as np
from qibo import Circuit
from qibo.backends import Backend
from qibo.config import raise_error

from qiboml import ndarray
from qiboml.backends.jax import JaxBackend
from qiboml.models.decoding import QuantumDecoding
from qiboml.models.encoding import DataReuploading, QuantumEncoding


@dataclass
class Differentiation(ABC):
    """
    Abstract differentiator object.
    """

    @abstractmethod
    def evaluate(
        self,
        x: ndarray,
        encoding: QuantumEncoding,
        training: Circuit,  # TODO: replace with an abstract TrainableLayer
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
            encoding (QunatumEncoding): the encoding layer.
            training (Circuit): the trainable quantum circuit.
            decoding (QunatumDecoding): the decoding layer.
            backend (Backend): the backend to execute the circuit with.
            parameters (list[ndarray]): the parameters at which to evaluate the model, and thus the derivative.
            wrt_inputs (bool): whether to calculate the derivate with respect to inputs or not, by default ``False``.

        Returns:
            (list[ndarray]): the calculated gradients.
        """
        pass

    def _full_circuit(
        self,
        x: ndarray,
        encoding: QuantumEncoding,
        training: Union[Circuit, List[Circuit]],
    ):
        """
        Helper method to reconstruct the full circuit to cover the reuploading
        strategy.
        """
        circ = Circuit(encoding.nqubits)
        if isinstance(encoding, DataReuploading):
            for l in range(encoding.nlayers):
                circ += encoding(x) + training[l]
        else:
            circ += encoding(x) + training
        return circ


class PSR(Differentiation):
    """
    The Parameter Shift Rule differentiatior. Especially useful for non analytical
    derivative calculation which, thus, makes it hardware compatible.
    """

    def evaluate(
        self,
        x: ndarray,
        encoding: QuantumEncoding,
        training: Circuit,  # TODO: replace with an abstract TrainableLayer
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
            encoding (QunatumEncoding): the encoding layer.
            training (Circuit): the trainable quantum circuit.
            decoding (QunatumDecoding): the decoding layer.
            backend (Backend): the backend to execute the circuit with.
            parameters (list[ndarray]): the parameters at which to evaluate the model, and thus the derivative.
            wrt_inputs (bool): whether to calculate the derivate with respect to inputs or not, by default ``False``.


        Returns:
            (list[ndarray]): the calculated gradients.
        """
        if decoding.output_shape != (1, 1):
            raise_error(
                NotImplementedError,
                "Parameter Shift Rule only supports expectation value decoding.",
            )
        # construct circuit
        circuit = self._full_circuit(x=x, encoding=encoding, training=training)

        gradient = []
        if wrt_inputs:
            # compute first gradient part, wrt data
            gradient.append(
                backend.np.reshape(
                    backend.np.hstack(
                        self.gradient_wrt_inputs(
                            x,
                            encoding,
                            circuit,
                            decoding,
                        )
                    ),
                    (decoding.output_shape[-1], x.shape[-1]),
                )
            )
        else:
            gradient.append(backend.np.zeros((decoding.output_shape[-1], x.shape[-1])))

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

        tmp_params = backend.cast(parameters, copy=True)
        tmp_params = self.shift_parameter(tmp_params, parameter_index, s, backend)

        circuit.set_parameters(tmp_params)
        forward = decoding(circuit)

        tmp_params = self.shift_parameter(tmp_params, parameter_index, -2 * s, backend)

        circuit.set_parameters(tmp_params)
        backward = decoding(circuit)
        return generator_eigenval * (forward - backward)

    def gradient_wrt_inputs(
        self,
        x,
        encoding,
        circuit,
        decoding,
    ):
        if isinstance(encoding, DataReuploading):
            raise NotImplementedError(
                f"DataReuploading encoding does not support derivative wrt inputs yet."
            )
        gates = encoding(x).queue
        gradient = []
        for input, gate in zip(x, gates):
            shift = np.pi / (4 * gate.generator_eigenvalue())
            forward = encoding(input + shift) + circuit
            backward = encoding(input - shift) + circuit
            gradient.append(decoding(forward) - decoding(backward))
        return gradient

    @staticmethod
    def shift_parameter(parameters, i, epsilon, backend):
        if backend.platform == "tensorflow":
            return backend.tf.stack(
                [parameters[j] + int(i == j) * epsilon for j in range(len(parameters))]
            )
        elif backend.platform == "jax":
            parameters.at[i].set(parameters[i] + epsilon)
        else:
            parameters[i] = parameters[i] + epsilon
        return parameters


class Jax(Differentiation):
    """
    The Jax differentiator object. Particularly useful for enabling gradient calculation in
    those backends that do not provide it. Note however, that for this reason the circuit is
    executed with the JaxBackend whenever a derivative is needed
    """

    def __init__(self):
        self._jax: Backend = JaxBackend()
        self._argnums: tuple[int] = None

    def evaluate(
        self,
        x: ndarray,
        encoding: QuantumEncoding,
        circuit: Circuit,  # TODO: replace with an abstract TrainableLayer
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
            encoding (QunatumEncoding): the encoding layer.
            training (Circuit): the trainable quantum circuit.
            decoding (QunatumDecoding): the decoding layer.
            backend (Backend): the backend to execute the circuit with.
            parameters (list[ndarray]): the parameters at which to evaluate the model, and thus the derivative.
            wrt_inputs (bool): whether to calculate the derivate with respect to inputs or not, by default ``False``.


        Returns:
            (list[ndarray]): the calculated gradients.
        """
        x = backend.to_numpy(x)
        x = self._jax.cast(x, self._jax.precision)
        if self._argnums is None:
            self._argnums = tuple(range(4, len(parameters) + 4))
            setattr(
                self,
                "_jacobian",
                partial(jax.jit, static_argnums=(1, 2, 3))(
                    jax.jacfwd(self._run, (0,) + self._argnums),
                ),
            )
            setattr(
                self,
                "_jacobian_without_inputs",
                partial(jax.jit, static_argnums=(1, 2, 3))(
                    jax.jacfwd(self._run, self._argnums),
                ),
            )
        parameters = backend.to_numpy(list(parameters))
        parameters = self._jax.cast(parameters, parameters.dtype)
        decoding.set_backend(self._jax)
        if wrt_inputs:
            gradients = self._jacobian(  # pylint: disable=no-member
                x, encoding, circuit, decoding, *parameters
            )
        else:
            gradients = (
                self._jax.numpy.zeros((decoding.output_shape[-1], x.shape[-1])),
                self._jacobian_without_inputs(  # pylint: disable=no-member
                    x, encoding, circuit, decoding, *parameters
                ),
            )
        decoding.set_backend(backend)
        return [
            backend.cast(self._jax.to_numpy(grad).tolist(), backend.precision)
            for grad in gradients
        ]

    @staticmethod
    @partial(jax.jit, static_argnums=(1, 2, 3))
    def _run(x, encoding, training, decoding, *parameters):
        # Use the shared full_circuit method from Differentiation
        circ = Differentiation.full_circuit(x, encoding, training)
        circ.set_parameters(parameters)
        return decoding(circ)
