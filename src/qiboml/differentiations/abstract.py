from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property, partial
from typing import Callable, Optional

import numpy as np
from numpy.typing import ArrayLike
from qibo import Circuit
from qibo.backends import construct_backend

from qiboml.models.decoding import QuantumDecoding


@dataclass
class Differentiation(ABC):
    """
    Abstract differentiator object.
    """

    circuit: Optional[Circuit] = None
    decoding: Optional[QuantumDecoding] = None
    _is_built: bool = False

    def __post_init__(self):
        if self.circuit is not None and self.decoding is not None:
            self.build(self.circuit, self.decoding)

    def build(self, circuit: Circuit, decoding: QuantumDecoding):
        """Attach model internals and prepare compiled artifacts."""
        if self._is_built:  # pragma: no cover
            return
        self.circuit = circuit
        self.decoding = decoding
        self._on_build()
        self._is_built = True

    def _on_build(self) -> None:
        pass

    @abstractmethod
    def evaluate(
        self, parameters: ArrayLike, wrt_inputs: bool = False
    ):  # pragma: no cover
        """
        Evaluate the gradient of the quantum circuit w.r.t its parameters, i.e. its rotation angles.
        """
        pass

    @property
    def backend(self):
        assert self.decoding is not None, "Differentiator not built yet."
        return self.decoding.backend

    @cached_property
    def non_trainable_gates(self):
        assert self.circuit is not None, "Differentiator not built yet."
        return [g for g in self.circuit.parametrized_gates if not g.trainable]

    def nparams(self, wrt_inputs):
        assert self.circuit is not None, "Differentiator not built yet."
        return len(self.circuit.get_parameters(include_not_trainable=wrt_inputs))


# class Jax(Differentiation):

#     def __post_init__(self):
#         super().__post_init__()
#         self._jax = construct_backend("qiboml", platform="jax")

#     def _on_build(self):
#         self._compile_jacobians()

#     def _compile_jacobians(self):
#         n_params = len(
#             [
#                 p
#                 for params in self.circuit.get_parameters(include_not_trainable=False)
#                 for p in params
#             ]
#         )
#         n_outputs = int(np.prod(self.decoding.output_shape))
#         jac = jax.jacfwd if n_params < n_outputs else jax.jacrev
#         self._jacobian: Callable = partial(jax.jit, static_argnums=(0, 1))(
#             jac(self._run, tuple(range(2, n_params + 2))),
#         )
#         n_params = len(
#             [
#                 p
#                 for params in self.circuit.get_parameters(include_not_trainable=True)
#                 for p in params
#             ]
#         )
#         jac = jax.jacfwd if n_params < n_outputs else jax.jacrev
#         self._jacobian_with_inputs: Callable = partial(jax.jit, static_argnums=(0, 1))(
#             jac(self._run_with_inputs, tuple(range(2, n_params + 2))),
#         )

#     @staticmethod
#     @partial(jax.jit, static_argnums=(0, 1))
#     def _run(circuit, decoding, *parameters):
#         for g, p in zip(circuit.trainable_gates, parameters):
#             g.parameters = p
#         circuit._final_state = None
#         return decoding(circuit)

#     @staticmethod
#     @partial(jax.jit, static_argnums=(0, 1))
#     def _run_with_inputs(circuit, decoding, *parameters):
#         for g, p in zip(circuit.parametrized_gates, parameters):
#             g.parameters = p
#         circuit._final_state = None
#         return decoding(circuit)

#     def _cast_non_trainable_parameters(self, src_backend, tgt_backend):
#         for g in self.non_trainable_gates:
#             g.parameters = tgt_backend.cast(
#                 np.array([src_backend.to_numpy(par) for par in g.parameters]),
#                 dtype=tgt_backend.float64,
#             )

#     def evaluate(self, parameters: list[ArrayLike], wrt_inputs: bool = False):
#         """
#         Evaluate the jacobian of the internal quantum model (circuit + decoding) w.r.t to its ``parameters``,
#         *i.e.* the parameterized gates in the circuit.

#         Args:
#             parameters (list[ArrayLike]): the parameters at which to evaluate the circuit,
#                 and thus the derivatives.
#             wrt_inputs (bool): whether to calculate the derivative with respect to,
#                 also, inputs (i.e. encoding angles) or not. Defaults to ``False``.

#         Returns:
#             ArrayLike: The calculated jacobian.
#         """

#         assert (
#             self._is_built
#         ), "Call .build_differentiation(circuit, decoding) before evaluate()."

#         # backup the backend
#         backend = self.decoding.backend
#         # convert params to jax
#         params = np.array(backend.to_numpy(parameters))
#         params = self._jax.cast(params, dtype=self._jax.float64)
#         if not wrt_inputs:
#             self._cast_non_trainable_parameters(self.backend, self._jax)
#         # set jax for running
#         self.decoding.set_backend(self._jax)
#         # calculate the jacobian
#         jac_f = self._jacobian_with_inputs if wrt_inputs else self._jacobian
#         jacobian = jac_f(  # pylint: disable=not-callable
#             self.circuit, self.decoding, *params
#         )
#         # reset the original backend
#         self.decoding.set_backend(backend)
#         # reset the original parameters
#         target_gates = (
#             self.circuit.parametrized_gates
#             if wrt_inputs
#             else self.circuit.trainable_gates
#         )
#         for g, p in zip(target_gates, parameters):
#             g.parameters = p
#         if not wrt_inputs:
#             self._cast_non_trainable_parameters(self._jax, self.backend)
#         self.circuit._final_state = None

#         # transform back to the backend native array
#         return backend.cast(self._jax.to_numpy(jacobian).tolist(), backend.float64)
