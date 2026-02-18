import math
from typing import Optional, Callable

import jax
from qibo import Circuit
from qibo.backends import construct_backend
from qiboml.differentiations.jax import Jax
from qiboml.models.decoding import QuantumDecoding


class QuimbJax(Jax):  # pragma: no cover

    def __init__(
        self,
        circuit: Optional[Circuit] = None,
        decoding: Optional[QuantumDecoding] = None,
        _is_built: bool = False,
        **quimb_kwargs
    ):
        self.quimb_kwargs = quimb_kwargs
        super().__init__(circuit, decoding)

    def __post_init__(self):
        super().__post_init__()
        self._jax = construct_backend(
            "qibotn",
            platform="quimb",
            quimb_backend="jax",
            contraction_optimizer=self.quimb_kwargs.get(
                "contraction_optimizer", "auto-hq"
            ),
        )
        self._jax.configure_tn_simulation(
            self.quimb_kwargs.get("ansatz", "mps"),
            self.quimb_kwargs.get("max_bond_dimension", None),
            self.quimb_kwargs.get("svd_cutoff", 1e-10),
            self.quimb_kwargs.get("n_most_frequent_states", 100),
        )

    def _compile_jacobians(self):
        n_params = len(
            [
                p
                for params in self.circuit.get_parameters(include_not_trainable=False)
                for p in params
            ]
        )
        n_outputs = int(math.prod(self.decoding.output_shape))
        jac = jax.jacfwd if n_params < n_outputs else jax.jacrev
        self._jacobian: Callable = jac(self._run, tuple(range(2, n_params + 2)))

        n_params = len(
            [
                p
                for params in self.circuit.get_parameters(include_not_trainable=True)
                for p in params
            ]
        )
        jac = jax.jacfwd if n_params < n_outputs else jax.jacrev
        self._jacobian_with_inputs: Callable = jac(
            self._run_with_inputs, tuple(range(2, n_params + 2))
        )

    @staticmethod
    def _run(circuit: Circuit, decoding: QuantumDecoding, *parameters):
        for g, p in zip(circuit.trainable_gates, parameters):
            g.parameters = p
        circuit._final_state = None
        return decoding(circuit)

    @staticmethod
    def _run_with_inputs(circuit: Circuit, decoding: QuantumDecoding, *parameters):
        for g, p in zip(circuit.parametrized_gates, parameters):
            g.parameters = p
        circuit._final_state = None
        return decoding(circuit)
