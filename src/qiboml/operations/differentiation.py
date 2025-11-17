from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property, partial
from typing import Callable, Tuple

import jax
import numpy as np
from qibo import Circuit
from qibo.backends import Backend
from qibo.config import raise_error

from qiboml import ndarray
from qiboml.backends.jax import JaxBackend
from qiboml.models.decoding import Expectation, QuantumDecoding


@dataclass
class Differentiation(ABC):
    """
    Abstract differentiator object.
    """

    circuit: Circuit
    decoding: QuantumDecoding

    @abstractmethod
    def evaluate(
        self, parameters: ndarray, wrt_inputs: bool = False
    ):  # pragma: no cover
        """
        Evaluate the gradient of the quantum circuit w.r.t its parameters, i.e. its rotation angles.
        """
        pass

    @property
    def backend(self):
        return self.decoding.backend

    @cached_property
    def non_trainable_gates(self):
        return [g for g in self.circuit.parametrized_gates if not g.trainable]

    def nparams(self, wrt_inputs):
        return len(self.circuit.get_parameters(include_not_trainable=wrt_inputs))


@dataclass
class PSR(Differentiation):
    """
    The Parameter Shift Rule differentiator. Especially useful for non analytical
    derivative calculation which, thus, makes it hardware compatible.
    """

    def __post_init__(self):
        if np.prod(self.decoding.output_shape) != 1:
            raise_error(
                RuntimeError,
                "PSR differentiation works only for decoders with scalar outpus, i.e. expectation values.",
            )

    def evaluate(self, parameters: ndarray, wrt_inputs: bool = False):
        """
        Evaluate the gradient of the quantum circuit w.r.t its parameters, i.e. its rotation angles.
        Args:
            parameters (List[ndarray]): the parameters at which to evaluate the model, and thus the derivative.
            wrt_inputs (bool): whether to calculate the derivative with respect to, also, inputs (i.e. encoding angles)
        or not, by default ``False``.
        Returns:
            (ndarray): the calculated jacobian.
        """

        circuits = []
        eigvals = []

        for i in range(self.nparams(wrt_inputs)):
            forward, backward, eigval = self.one_parameter_shift(
                parameters=parameters, parameter_index=i, wrt_inputs=wrt_inputs
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
        self, parameters: ndarray, parameter_index: int, wrt_inputs: bool = False
    ) -> Tuple[Circuit, Circuit, float]:
        """Compute one derivative of the decoding strategy w.r.t. a target parameter."""
        target_gates = (
            self.circuit.parametrized_gates
            if wrt_inputs
            else self.circuit.trainable_gates
        )
        gate = target_gates[parameter_index]
        generator_eigenval = gate.generator_eigenvalue()
        s = np.pi / (4 * generator_eigenval)

        tmp_params = self.backend.cast(parameters, copy=True, dtype=parameters[0].dtype)
        tmp_params = self.shift_parameter(tmp_params, parameter_index, s, self.backend)

        forward = self.circuit.copy(True)
        target_gates = (
            forward.parametrized_gates if wrt_inputs else forward.trainable_gates
        )
        # forward.set_parameters(tmp_params)
        for g, p in zip(target_gates, tmp_params):
            g.parameters = p

        tmp_params = self.backend.cast(parameters, copy=True, dtype=parameters[0].dtype)
        tmp_params = self.shift_parameter(tmp_params, parameter_index, -s, self.backend)

        backward = self.circuit.copy(True)
        target_gates = (
            backward.parametrized_gates if wrt_inputs else backward.trainable_gates
        )
        # backward.set_parameters(tmp_params)
        for g, p in zip(target_gates, tmp_params):
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


class Adjoint(Differentiation):
    """
    Adjoint differentiation following Algorithm 1 from https://arxiv.org/abs/2009.02823.
    Only :class:`qiboml.models.decoding.Expectation`. as decoding is supported and all parametrized_gates
    must have a gradient method returning the gradient of the single gate.
    """

    def evaluate(self, parameters: ndarray, wrt_inputs: bool = False):
        """
        Evaluate the gradient of the quantum circuit w.r.t its parameters, i.e. its rotation angles.
        Args:
            parameters (List[ndarray]): the parameters at which to evaluate the model, and thus the derivative.
            wrt_inputs (bool): whether to calculate the derivate with respect to inputs or not, by default ``False``.
        Returns:
            (ndarray): the calculated gradients.
        """
        assert isinstance(
            self.decoding, Expectation
        ), "Adjoint differentation supported only for Expectation."
        gate_list = (
            self.circuit.trainable_gates
            if not wrt_inputs
            else self.circuit.parametrized_gates
        )
        for g, p in zip(gate_list, parameters):
            g.parameters = p

        gradients = []
        lam = self.backend.execute_circuit(self.circuit).state()
        nqubits = self.circuit.nqubits
        phi = lam
        lam = self.decoding.observable @ lam  # pylint: disable=E1101
        for gate in reversed(self.circuit.queue):
            phi = self.backend.apply_gate(gate.dagger(), phi, nqubits=nqubits)
            if gate in gate_list:
                mu = phi
                mu = self.backend.apply_gate(
                    gate.gradient(backend=self.backend), mu, nqubits=nqubits
                )
                gradients.append(
                    2 * self.backend.np.real(self.backend.np.vdot(lam, mu))
                )
            lam = self.backend.apply_gate(gate.dagger(), lam, nqubits=nqubits)
        return self.backend.cast(gradients[::-1], dtype=parameters.dtype).reshape(
            -1, *self.decoding.output_shape
        )


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
                for params in self.circuit.get_parameters(include_not_trainable=False)
                for p in params
            ]
        )
        n_outputs = int(np.prod(self.decoding.output_shape))
        jac = jax.jacfwd if n_params < n_outputs else jax.jacrev
        self._jacobian: Callable = partial(jax.jit, static_argnums=(0, 1))(
            jac(self._run, tuple(range(2, n_params + 2))),
        )
        n_params = len(
            [
                p
                for params in self.circuit.get_parameters(include_not_trainable=True)
                for p in params
            ]
        )
        jac = jax.jacfwd if n_params < n_outputs else jax.jacrev
        self._jacobian_with_inputs: Callable = partial(jax.jit, static_argnums=(0, 1))(
            jac(self._run_with_inputs, tuple(range(2, n_params + 2))),
        )

    @staticmethod
    @partial(jax.jit, static_argnums=(0, 1))
    def _run(circuit, decoding, *parameters):
        for g, p in zip(circuit.trainable_gates, parameters):
            g.parameters = p
        # circuit.set_parameters(parameters)
        return decoding(circuit)

    @staticmethod
    @partial(jax.jit, static_argnums=(0, 1))
    def _run_with_inputs(circuit, decoding, *parameters):
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

    def _cast_non_trainable_parameters(self, src_backend, tgt_backend):
        for g in self.non_trainable_gates:
            g.parameters = tgt_backend.cast(
                np.array([src_backend.to_numpy(par) for par in g.parameters]),
                dtype=tgt_backend.np.float64,
            )

    def evaluate(self, parameters, wrt_inputs: bool = False):
        """
        Evaluate the jacobian of the internal quantum model (circuit + decoding) w.r.t to its ``parameters``,
        i.e. the parameterized gates in the circuit.
        Args:
            parameters (list[ndarray]): the parameters at which to evaluate the circuit, and thus the derivatives.
            wrt_inputs (bool): whether to calculate the derivative with respect to, also, inputs (i.e. encoding angles)
        or not, by default ``False``.
        Returns:
            (ndarray): the calculated jacobian.
        """
        # backup the backend
        backend = self.decoding.backend
        # convert params to jax
        params = np.array(backend.to_numpy(parameters))
        params = self._jax.cast(params, dtype=self._jax.np.float64)
        if not wrt_inputs:
            self._cast_non_trainable_parameters(self.backend, self._jax)
        # set jax for running
        self.decoding.set_backend(self._jax)
        # calculate the jacobian
        jac_f = self._jacobian_with_inputs if wrt_inputs else self._jacobian
        jacobian = jac_f(  # pylint: disable=not-callable
            self.circuit, self.decoding, *params
        )
        # reset the original backend
        self.decoding.set_backend(backend)
        # reset the original parameters
        target_gates = (
            self.circuit.parametrized_gates
            if wrt_inputs
            else self.circuit.trainable_gates
        )
        for g, p in zip(target_gates, parameters):
            g.parameters = p
        if not wrt_inputs:
            self._cast_non_trainable_parameters(self._jax, self.backend)
        # transform back to the backend native array
        return backend.cast(self._jax.to_numpy(jacobian).tolist(), backend.np.float64)
