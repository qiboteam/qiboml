from typing import Optional

from qibo.backends import _check_backend
from qibo.backends.abstract import Backend
from qibo.quantum_info.metrics import quantum_fisher_information_matrix


class QuantumNaturalGradient:
    def __init__(
        self, learning_rate: Optional[float] = 1e-3, backend: Optional[Backend] = None
    ):
        self.backend = _check_backend(backend)
        self.learning_rate = learning_rate

    def gradients(self, params, **kwargs):
        return self.backend.calculate_gradients(params, **kwargs)

    def update_parameters(self, grads, params, **kwargs):
        circuit = kwargs["circuit"]

        inverse_metric = kwargs.get("inverse_metric", None)

        if inverse_metric is None:
            inverse_metric = quantum_fisher_information_matrix(
                circuit, params, backend=self.backend
            )
            inverse_metric = self.backend.calculate_matrix_inverse(inverse_metric)

        if inverse_metric is not None and len(inverse_metric.shape) == 1:
            natural_gradient = inverse_metric * grads
        else:
            natural_gradient = inverse_metric @ grads

        self.backend.apply_gradients(natural_gradient, params, self.learning_rate)
