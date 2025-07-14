"""Module with the implementation of the quantum natural gradient."""

from typing import Optional

import tensorflow as tf
from qibo import Circuit
from qibo.backends import Backend, _check_backend
from qibo.quantum_info.metrics import quantum_fisher_information_matrix

from qiboml.backends.tensorflow import TensorflowBackend


class QuantumNaturalGradientTensorflow:
    def __init__(
        self,
        circuit: Circuit,
        learning_rate: Optional[float] = 1e-3,
        backend: Optional[Backend] = None,
        **kwargs,
    ):
        self.backend = _check_backend(backend)
        assert isinstance(self.backend, TensorflowBackend)

        self.circuit = circuit
        self.learning_rate = learning_rate

    def gradients(self, circuit, params):
        with self.backend.tf.GradientTape() as tape:
            tape.watch(params)
            circuit.set_parameters(params)
            result = self.backend.execute_circuit(circuit)
        return tape.gradient(result.state(), params)

    def update_parameters(self, grads, params):
        inv_metric = quantum_fisher_information_matrix(
            self.circuit, params[0], backend=self.backend
        )
        inv_metric = self.backend.tf.linalg.inv(inv_metric)

        natural_grad = inv_metric @ grads[0]

        params[0].assign_sub(self.learning_rate * natural_grad)
