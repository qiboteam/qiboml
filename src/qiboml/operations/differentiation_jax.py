from functools import partial
from typing import Callable

import numpy as np
import jax  # pylint: disable=import-error
from qibo import Circuit
from qibo.backends.abstract import Backend
from qiboml.backends.jax import JaxBackend
from qiboml.models.decoding import QuantumDecoding
from qiboml.operations.differentiation import Differentiation


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
                np.array([src_backend.to_numpy(par) for par in g.parameters])
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
        params = self._jax.cast(params, dtype=self._jax.engine.float64)
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
        return backend.cast(self._jax.to_numpy(jacobian).tolist(), backend.float64)
