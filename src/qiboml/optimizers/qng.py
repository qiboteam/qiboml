"""Module with the implementation of the quantum natural gradient."""

from typing import List, Optional

import torch
from qibo.config import raise_error
from qibo.quantum_info.metrics import quantum_fisher_information_matrix
from torch import Tensor
from torch.optim import SGD
from torch.optim.optimizer import (
    _default_to_fused_or_foreach,
    _use_grad_for_differentiable,
)
from torch.optim.sgd import _fused_sgd, _multi_tensor_sgd

from qiboml.backends.pytorch import PyTorchBackend


class QuantumNaturalGradient(SGD):
    def __init__(self, params, circuit, lr: float = 1e-3, **kwargs):
        kwargs["lr"] = lr

        super().__init__(params, **kwargs)

        self._circuit = circuit

    @_use_grad_for_differentiable
    def step(self, closure=None):
        """Perform a single optimization step with custom gradient modification."""

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params: List[Tensor] = []
            grads: List[Tensor] = []
            momentum_buffer_list: List[Optional[Tensor]] = []

            _ = self._init_group(group, params, grads, momentum_buffer_list)

            _single_tensor_qng(
                params,
                circuit=self._circuit,
                grads=grads,
                weight_decay=group["weight_decay"],
                lr=group["lr"],
                maximize=group["maximize"],
                grad_scale=getattr(self, "grad_scale", None),
                found_inf=getattr(self, "found_inf", None),
            )

        return loss


def _single_tensor_qng(
    params: List[Tensor],
    circuit,
    grads: List[Tensor],
    grad_scale: Optional[Tensor],
    found_inf: Optional[Tensor],
    *,
    weight_decay: float,
    lr: float,
    maximize: bool,
):
    assert grad_scale is None and found_inf is None

    for k, param in enumerate(params):
        grad = -grads[k] if maximize else grads[k]

        if weight_decay != 0:
            grad = grad.add(param, alpha=weight_decay)

        metric = quantum_fisher_information_matrix(
            circuit, param, backend=PyTorchBackend()
        )
        inv_metric = torch.linalg.inv(metric)
        raised_grad = inv_metric @ grad

        param.add_(raised_grad, alpha=-lr)
