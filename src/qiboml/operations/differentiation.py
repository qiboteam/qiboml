from abc import ABC, abstractmethod
from copy import deepcopy
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
class DifferentiationRule(ABC):

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
        Evaluate the gradient of a quantum model w.r.t variables and parameters,
        respectively represented by `x` and `parameters`.
        """
        pass


class PSR(DifferentiationRule):
    """
    Compute the gradient of the expectation value of a target observable w.r.t
    features and parameters contained in a quantum model using the parameter shift
    rules.
    """

    def evaluate(self, x: ndarray, encoding, training, decoding, backend, *parameters):
        if decoding.output_shape != (1, 1):
            raise_error(
                NotImplementedError,
                "Parameter Shift Rule only supports expectation value decoding.",
            )
        # construct circuit
        circuit = encoding(x) + training

        # compute first gradient part, wrt data
        gradient = self.gradient_wrt_data(
            data=x,
            encoding=encoding,
            circuit=circuit,
            decoding=decoding,
            backend=backend,
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

        tmp_params = backend.cast(parameters, copy=True)
        tmp_params = self.shift_parameter(tmp_params, parameter_index, s, backend)

        circuit.set_parameters(tmp_params)
        forward = decoding(circuit)

        tmp_params = self.shift_parameter(tmp_params, parameter_index, -2 * s, backend)

        circuit.set_parameters(tmp_params)
        backward = decoding(circuit)
        return generator_eigenval * (forward - backward)

    def gradient_wrt_data(
        self,
        data,
        encoding,
        circuit,
        decoding,
        backend,
    ):
        """
        Compute the gradient w.r.t. data.

        Args:
            data: data;
            encoding: encoding part of the quantum model. It is used to check whether
                parameter shift rules can be used to compute the gradient.
            circuit: all the quantum circuit, composed of encoding + eventual trainable
                layer.
            decoding: decoding part of the quantum model. In the PSR the decoding
                is usually a qiboml.models.decoding.Expectation layer.
            backend: qibo backend on which the circuit execution is performed-
        """
        x_size = backend.to_numpy(data).size
        # what follows now works for encodings in which the angle is equal to the feature
        # TODO: adapt this strategy to the more general case of a callable(x, params)
        if encoding.hardware_differentiable:
            x_gradient = []
            # loop over data components
            for k in range(x_size):
                # initialize derivative
                derivative_k = 0.0
                # extract gates which are encoding component x_k
                gates_encoding_xk = encoding.gates_encoding_feature(k)
                # loop over encoding gates
                for enc_gate in gates_encoding_xk:
                    # search for the target encoding gate in the circuit
                    generator_eigenval = enc_gate.generator_eigenvalue()
                    # TODO: the following shift value is valid only for rotation-like gates
                    shift = np.pi / (4 * generator_eigenval)
                    for gate in circuit.queue:
                        if gate == enc_gate:
                            original_parameter = deepcopy(gate.parameters)
                            gate.parameters = shifted_x_component(
                                x=data,
                                index=k,
                                shift_value=shift,
                                backend=backend,
                            )
                            forward = decoding(circuit)
                            gate.parameters = shifted_x_component(
                                x=data,
                                index=k,
                                shift_value=-2 * shift,
                                backend=backend,
                            )
                            backward = decoding(circuit)
                            derivative_k += float(
                                generator_eigenval * (forward - backward)
                            )
                            # restore original parameter
                            gate.parameters = original_parameter
                x_gradient.append(derivative_k)
            return [np.array([[[der for der in x_gradient]]])]

        else:
            # pad the gradients in case data are not uploaded into gates
            return [np.array([[(0.0,) * x_size]])]

    @staticmethod
    def shift_parameter(parameters, i, epsilon, backend):
        if backend.name == "tensorflow":
            return backend.tf.stack(
                [parameters[j] + int(i == j) * epsilon for j in range(len(parameters))]
            )
        elif backend.name == "jax":
            parameters.at[i].set(parameters[i] + epsilon)
        else:
            parameters[i] = parameters[i] + epsilon
        return parameters


class Jax(DifferentiationRule):

    def __init__(self):
        self._jax: Backend = JaxBackend()
        self._encoding = None
        self._training: Circuit = None
        self._decoding: QuantumDecoding = None
        self._argnums: list[int] = None
        self._circuit = None

    def evaluate(self, x: ndarray, encoding, training, decoding, backend, *parameters):
        binary = isinstance(encoding, BinaryEncoding)
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
        if binary:
            self._circuit = encoding(x) + training
        else:
            self._encoding = encoding
            self._training = training
        self._decoding = decoding
        self._decoding.set_backend(self._jax)
        if binary:
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


def shifted_x_component(
    x: ndarray, index: int, shift_value: float, backend: Backend
) -> float:
    """Shift a component of an ndarray."""
    flat_array = backend.to_numpy(x).flatten()
    shifted_flat_array = deepcopy(flat_array)
    shifted_flat_array[index] += shift_value
    return shifted_flat_array[index]
