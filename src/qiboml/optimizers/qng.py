"""Module with the implementation of the quantum natural gradient."""

from typing import List, Optional

import tensorflow as tf
import torch
from qibo.backends import _check_backend
from qibo.quantum_info.metrics import quantum_fisher_information_matrix
from torch import Tensor
from torch.optim import SGD
from torch.optim.optimizer import _use_grad_for_differentiable

from qiboml.backends.pytorch import PyTorchBackend
from qiboml.backends.tensorflow import TensorflowBackend


class _QuantumNaturalGradientTensorflow(tf.Module):
    def __init__(self, learning_rate: float = 1e-3, **kwargs):
        self.learning_rate = learning_rate

        self._circuit = kwargs["circuit"]

        self.backend = kwargs.get("backend", None)
        self.backend = _check_backend(self.backend)

    def apply_gradients(self, grads, params):
        inv_metric = quantum_fisher_information_matrix(
            self._circuit, params[0], backend=TensorflowBackend()
        )
        inv_metric = tf.linalg.inv(inv_metric)

        natural_grad = inv_metric @ grads[0]

        params[0].assign_sub(self.learning_rate * natural_grad)


class _QuantumNaturalGradientTorch(SGD):
    def __init__(self, circuit, learning_rate: float = 1e-3, **kwargs):
        kwargs["lr"] = learning_rate

        super().__init__(**kwargs)

        self._circuit = circuit

        self.backend = kwargs.get("backend", None)
        self.backend = _check_backend(self.backend)

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

            self._single_tensor_qng(
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
        self,
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
                circuit, param, backend=self.backend
            )
            inv_metric = torch.linalg.inv(metric)
            raised_grad = inv_metric @ grad

            param.add_(raised_grad, alpha=-lr)


class QuantumNaturalGradient(
    _QuantumNaturalGradientTorch, _QuantumNaturalGradientTensorflow
):

    def __init__(self, circuit, learning_rate: float = 1e-3, backend=None, **kwargs):
        backend = _check_backend(backend)

        if isinstance(backend, PyTorchBackend):
            params = kwargs["params"]
            _QuantumNaturalGradientTorch.__init__(
                self, params, circuit=circuit, lr=learning_rate, **kwargs
            )
        elif isinstance(backend, TensorflowBackend):
            _QuantumNaturalGradientTensorflow.__init__(
                self,
                learning_rate=learning_rate,
                circuit=circuit,
            )

        # return _QuantumNaturalGradientTensorflow(params, circuit, lr, **kwargs)
