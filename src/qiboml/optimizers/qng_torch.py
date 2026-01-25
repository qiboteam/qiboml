"""Module with the implementation of the quantum natural gradient."""

from typing import Optional

from qibo import Circuit
from qibo.backends import Backend, _check_backend
from qibo.quantum_info.metrics import quantum_fisher_information_matrix
from torch.optim.optimizer import _use_grad_for_differentiable

from qiboml.backends.pytorch import PyTorchBackend

# from qiboml.optimizers.abstract import Optimizer


class QuantumNaturalGradientTorch:
    def __init__(
        self,
        circuit: Circuit,
        learning_rate: Optional[float] = 1e-3,
        backend: Optional[Backend] = None,
        **kwargs,
    ):

        self.backend = _check_backend(backend)
        assert isinstance(self.backend, PyTorchBackend)

        self.circuit = circuit
        self.learning_rate = learning_rate

        params = kwargs["params"]

        super().__init__(params, lr=self.learning_rate, **kwargs)

    def gradients(self, params, **kwargs):
        return [param.grad() for param in params if param.grad is not None]

    def update_parameters(self, grads, params, **kwargs):
        self.step(closure=kwargs.closure)

    @_use_grad_for_differentiable
    def step(self, closure=None):
        """Perform a single optimization step with custom gradient modification."""

        loss = None
        if closure is not None:
            with self.backend.np.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params = []
            grads = []
            for p in group["params"]:
                if p.grad is not None:
                    params.append(p)
                    grads.append(p.grad)

        for k, param in enumerate(params):
            grad = -grads[k] if maximize else grads[k]

            if weight_decay != 0:
                grad = grad.add(param, alpha=weight_decay)

            metric = quantum_fisher_information_matrix(
                circuit, param, backend=self.backend
            )
            inv_metric = self.backend.tf.linalg.inv(
                metric
            )  # pylint: disable=not-callable
            raised_grad = inv_metric @ grad

            param.add_(raised_grad, alpha=-self.learning_rate)

    #         self._single_tensor_qng(
    #             params,
    #             circuit=self.circuit,
    #             grads=grads,
    #             weight_decay=group["weight_decay"],
    #             lr=group["lr"],
    #             maximize=group["maximize"],
    #             grad_scale=getattr(self, "grad_scale", None),
    #             found_inf=getattr(self, "found_inf", None),
    #         )

    #     return loss

    # def _single_tensor_qng(
    #     self,
    #     params: list,
    #     circuit: Circuit,
    #     grads: list,
    #     *,
    #     weight_decay: float,
    #     lr: float,
    #     maximize: bool,
    #     grad_scale=None,
    #     found_inf=None,
    # ):
    #     assert grad_scale is None and found_inf is None
