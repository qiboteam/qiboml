from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property
from typing import Tuple


import numpy as np
from qibo import Circuit
from qibo.config import raise_error

from qiboml import ndarray
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
        eigvals = self.backend.reshape(
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
            return backend.engine.stack(
                [parameters[j] + int(i == j) * epsilon for j in range(len(parameters))]
            )
        
        if backend.platform == "jax":
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
                    2 * self.backend.real(self.backend.engine.vdot(lam, mu))
                )
            lam = self.backend.apply_gate(gate.dagger(), lam, nqubits=nqubits)
        return self.backend.cast(gradients[::-1], dtype=parameters.dtype).reshape(
            -1, *self.decoding.output_shape
        )
