from abc import ABC, abstractmethod
from dataclasses import dataclass

import jax
import numpy as np
from qibo import Circuit
from qibo.backends import Backend
from qibo.config import raise_error

from qiboml import ndarray
from qiboml.backends.jax import JaxBackend
from qiboml.models.decoding import QuantumDecoding
from qiboml.models.encoding import BinaryEncoding, QuantumEncoding


@dataclass
class Differentiation(ABC):

    @abstractmethod
    def evaluate(
        self,
        x: ndarray,
        encoding: QuantumEncoding,
        training: Circuit,  # TODO: replace with an abstract TrainableLayer
        decoding: QuantumDecoding,
        backend: Backend,
        *parameters: ndarray,
    ):
        """
        Evaluate the gradient of a quantum model w.r.t inputs and parameters,
        respectively represented by `x` and `parameters`.
        """
        pass


class PSR(Differentiation):
    """
    Compute the gradient of the expectation value of a target observable w.r.t
    features and parameters contained in a quantum model using the parameter shift
    rules.
    """

    def evaluate(
        self,
        x: ndarray,
        encoding,
        training,
        decoding,
        backend,
        *parameters,
        wrt_inputs: bool = False,
    ):
        if decoding.output_shape != (1, 1):
            raise_error(
                NotImplementedError,
                "Parameter Shift Rule only supports expectation value decoding.",
            )
        # construct circuit
        circuit = encoding(x) + training

        gradient = []
        if wrt_inputs:
            # compute first gradient part, wrt data
            gradient.append(
                self.gradient_wrt_inputs(
                    x,
                    encoding,
                    circuit,
                    decoding,
                    backend=backend,
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
        backend,
    ):
        shift = 1e-2
        gradient = []
        for input in x:
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

    def __init__(self):
        self._jax: Backend = JaxBackend()
        self._encoding = None
        self._training: Circuit = None
        self._decoding: QuantumDecoding = None
        self._argnums: list[int] = None
        self._circuit = None

    def evaluate(
        self,
        x: ndarray,
        encoding,
        training,
        decoding,
        backend,
        *parameters,
        wrt_inputs: bool = False,
    ):
        x = backend.to_numpy(x)
        x = self._jax.cast(x, self._jax.precision)
        if self._argnums is None:
            self._argnums = range(len(parameters) + 1)
            setattr(self, "_jacobian", jax.jit(jax.jacfwd(self._run, self._argnums)))
            setattr(
                self,
                "_jacobian_without_inputs",
                jax.jit(jax.jacfwd(self._run_without_inputs, self._argnums[:-1])),
            )
        parameters = backend.to_numpy(list(parameters))
        parameters = self._jax.cast(parameters, parameters.dtype)
        if not wrt_inputs:
            self._circuit = encoding(x) + training
        else:
            self._encoding = encoding
            self._training = training
        self._decoding = decoding
        self._decoding.set_backend(self._jax)
        if not wrt_inputs:
            gradients = (
                self._jax.numpy.zeros((decoding.output_shape[-1], x.shape[-1])),
                self._jacobian_without_inputs(*parameters),  # pylint: disable=no-member
            )
        else:
            gradients = self._jacobian(x, *parameters)  # pylint: disable=no-member
        decoding.set_backend(backend)
        return [
            backend.cast(self._jax.to_numpy(grad).tolist(), backend.precision)
            for grad in gradients
        ]

    def _run(self, x, *parameters):
        circuit = self._encoding(x) + self._training
        circuit.set_parameters(parameters)
        return self._decoding(circuit)

    def _run_without_inputs(self, *parameters):
        self._circuit.set_parameters(parameters)
        return self._decoding(self._circuit)
