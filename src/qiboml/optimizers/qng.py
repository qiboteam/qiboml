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


class QuantumNaturalGradient(SGD):
    def __init__(self, params, circuit, lr: float = 1e-3, **kwargs):
        kwargs["lr"] = lr
        super().__init__(params, kwargs)

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
            state_steps: List[Tensor] = []
            momentum_buffer_list: List[Optional[Tensor]] = []

            _qng(
                params,
                self._circuit,
                grads,
                state_steps,
                momentum_buffer_list,
                weight_decay=group["weight_decay"],
                momentum=group["momentum"],
                lr=group["lr"],
                dampening=group["dampening"],
                nesterov=group["nesterov"],
                maximize=group["maximize"],
                foreach=group["foreach"],
                fused=group["fused"],
                grad_scale=getattr(self, "grad_scale", None),
                found_inf=getattr(self, "found_inf", None),
                b=group["b"],
                lr_scheduling=group["lr_scheduling"],
            )

            if group["momentum"] != 0:
                # update momentum_buffers in state
                for p, momentum_buffer in zip(params, momentum_buffer_list):
                    state = self.state[p]
                    state["momentum_buffer"] = momentum_buffer

        return loss


def _qng(
    params: List[Tensor],
    circuit,
    d_p_list: List[Tensor],
    momentum_buffer_list: List[Optional[Tensor]],
    # kwonly args with defaults are not supported by functions compiled with torchscript issue #70627
    # setting this as kwarg for now as functional API is compiled by torch/distributed/optim
    has_sparse_grad: bool = False,
    foreach: Optional[bool] = None,
    fused: Optional[bool] = None,
    grad_scale: Optional[Tensor] = None,
    found_inf: Optional[Tensor] = None,
    *,
    weight_decay: float,
    momentum: float,
    lr: float,
    dampening: float,
    nesterov: bool,
    maximize: bool,
):
    if foreach is None and fused is None:
        # why must we be explicit about an if statement for torch.jit.is_scripting here?
        # because JIT can't handle Optionals nor fancy conditionals when scripting
        if not torch.jit.is_scripting():
            fused, foreach = _default_to_fused_or_foreach(
                params, differentiable=False, use_fused=False
            )
        else:
            foreach = False
            fused = False
    if foreach is None:
        foreach = False
    if fused is None:
        fused = False

    if foreach and torch.jit.is_scripting():
        raise RuntimeError("torch.jit.script not supported with foreach optimizers")
    if fused and torch.jit.is_scripting():
        raise RuntimeError("torch.jit.script not supported with fused optimizers")

    if foreach and not torch.jit.is_scripting():
        func = _multi_tensor_sgd
    elif fused and not torch.jit.is_scripting():
        func = _fused_sgd
    else:
        func = _single_tensor_qng

    func(
        params,
        circuit,
        d_p_list,
        momentum_buffer_list,
        weight_decay=weight_decay,
        momentum=momentum,
        lr=lr,
        dampening=dampening,
        nesterov=nesterov,
        has_sparse_grad=has_sparse_grad,
        maximize=maximize,
        grad_scale=grad_scale,
        found_inf=found_inf,
    )


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

    for i, param in enumerate(params):

        grad = grads[i] if not maximize else -grads[i]

        if weight_decay != 0:
            grad = grad.add(param, alpha=weight_decay)

        metric = quantum_fisher_information_matrix(circuit, param)
        inv_metric = torch.linalg.inv(metric)
        raised_grad = inv_metric @ grad

        params.add_(raised_grad, alpha=-lr)
