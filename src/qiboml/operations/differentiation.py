from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial
from typing import Callable, Tuple

import jax
import numpy as np
from qibo import Circuit
from qibo.backends import Backend
from qibo.config import raise_error
from tensorflow import no_op

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

    circuit: Circuit
    decoding: QuantumDecoding

    @abstractmethod
    def evaluate(
        self,
        parameters: ndarray,
    ):  # pragma: no cover
        """
        Evaluate the gradient of the quantum circuit w.r.t its parameters.
        """
        pass

    @property
    def backend(self):
        return self.decoding.backend


class PSR(Differentiation):
    """
    The Parameter Shift Rule differentiator. Especially useful for non analytical
    derivative calculation which, thus, makes it hardware compatible.
    """

    def evaluate(
        self,
        parameters: ndarray,
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
        circuits = []
        eigvals = []

        for i in range(len(parameters)):
            forward, backward, eigval = self.one_parameter_shift(
                parameters=parameters,
                parameter_index=i,
            )
            circuits.extend([forward, backward])
            eigvals.append(eigval)

        # TODO: parallelize when decoding will support
        # the parallel execution of multiple circuits
        expvals = self.backend.cast(
            [self.decoding(circ) for circ in circuits], dtype=parameters.dtype
        )
        forwards = expvals[::2]
        backwards = expvals[1::2]
        eigvals = self.backend.np.reshape(
            self.backend.cast(eigvals, dtype=parameters.dtype), forwards.shape
        )
        return (forwards - backwards) * eigvals

    def one_parameter_shift(
        self,
        parameters: ndarray,
        parameter_index: int,
    ) -> Tuple[Circuit, Circuit, float]:
        """Compute one derivative of the decoding strategy w.r.t. a target parameter."""
        gate = self.circuit.parametrized_gates[parameter_index]
        generator_eigenval = gate.generator_eigenvalue()
        s = np.pi / (4 * generator_eigenval)

        tmp_params = self.backend.cast(parameters, copy=True, dtype=parameters[0].dtype)
        tmp_params = self.shift_parameter(tmp_params, parameter_index, s, self.backend)

        forward = self.circuit.copy(True)
        # forward.set_parameters(tmp_params)
        for g, p in zip(forward.parametrized_gates, tmp_params):
            g.parameters = p

        tmp_params = self.backend.cast(parameters, copy=True, dtype=parameters[0].dtype)
        tmp_params = self.shift_parameter(tmp_params, parameter_index, -s, self.backend)

        backward = self.circuit.copy(True)
        # backward.set_parameters(tmp_params)
        for g, p in zip(backward.parametrized_gates, tmp_params):
            g.parameters = p

        return forward, backward, generator_eigenval

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

    def __init__(self, circuit: Circuit, decoding: QuantumDecoding):
        self._jax: Backend = JaxBackend()
        self._circuit = circuit
        self._decoding = decoding
        self.__post_init__()

    def __post_init__(self):
        n_params = len(
            [
                p
                for params in self.circuit.get_parameters(include_not_trainable=True)
                for p in params
            ]
        )
        n_outputs = int(np.prod(self.decoding.output_shape))
        jac = jax.jacfwd if n_params < n_outputs else jax.jacrev
        self._jacobian: Callable = partial(jax.jit, static_argnums=(0, 1))(
            jac(self._run, tuple(range(2, n_params + 2))),
        )

    @staticmethod
    @partial(jax.jit, static_argnums=(0, 1))
    def _run(circuit, decoding, *parameters):
        for g, p in zip(circuit.parametrized_gates, parameters):
            g.parameters = p
        # circuit.set_parameters(parameters)
        return decoding(circuit)

    @property
    def circuit(self):
        return self._circuit

    @circuit.setter
    def circuit(self, circ):
        self._circuit = circ
        self.__post_init__()

    @property
    def decoding(self):
        return self._decoding

    @decoding.setter
    def decoding(self, dec):
        self._decoding = dec
        self.__post_init__()

    def evaluate(self, parameters):
        """
        Evaluate the jacobian of the internal quantum model (circuit + decoding) w.r.t to its ``parameters``,
        i.e. the parameterized gates in the circuit.
        Args:
            parameters (list[ndarray]): the parameters at which to evaluate the circuit, and thus the derivatives.
        Returns:
            (ndarray): the calculated jacobian.
        """
        # backup the backend
        backend = self.decoding.backend
        # convert params to jax
        params = np.array(backend.to_numpy(parameters))
        params = self._jax.cast(params, dtype=self._jax.np.float64)
        # set jax for running
        self.decoding.set_backend(self._jax)
        # calculate the jacobian
        jacobian = self._jacobian(self.circuit, self.decoding, *params)
        # reset the original backend
        self.decoding.set_backend(backend)
        # reset the original parameters
        for g, p in zip(self.circuit.parametrized_gates, parameters):
            g.parameters = p
        # transform back to the backend native array
        return backend.cast(self._jax.to_numpy(jacobian).tolist(), backend.np.float64)
