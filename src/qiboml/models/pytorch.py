"""Torch interface to qiboml layers"""

from dataclasses import dataclass

import numpy as np
import torch
from qibo.backends import Backend
from qibo.config import raise_error

import qiboml.models.encoding_decoding as ed
from qiboml.models.abstract import QuantumCircuitLayer, _run_layers
from qiboml.operations import differentiation as Diff


@dataclass(eq=False)
class QuantumModel(torch.nn.Module):

    layers: list[QuantumCircuitLayer]
    differentiation: str = "PSR"

    def __post_init__(self):
        super().__init__()
        for layer in self.layers[1:]:
            if layer.circuit.nqubits != self.nqubits:
                raise_error(
                    RuntimeError,
                    f"Layer \n{layer}\n has {layer.circuit.nqubits} qubits, but {self.nqubits} qubits was expected.",
                )
            if layer.backend.name != self.backend.name:
                raise_error(
                    RuntimeError,
                    f"Layer \n{layer}\n is using {layer.backend} backend, but {self.backend} backend was expected.",
                )
        for layer in self.layers:
            if len(layer.parameters) > 0:
                params = (
                    layer.parameters
                    if self.backend.name == "pytorch"
                    else torch.as_tensor(np.array(layer.parameters))
                )
                self.register_parameter(
                    layer.__class__.__name__,
                    torch.nn.Parameter(params),
                )
        if not isinstance(self.layers[-1], ed.QuantumDecodingLayer):
            raise_error(
                RuntimeError,
                f"The last layer has to be a `QuantumDecodinglayer`, but is {self.layers[-1]}",
            )
        self.differentiation = getattr(Diff, self.differentiation)(self.backend)

    def forward(self, x: torch.Tensor):
        if self.backend.name != "pytorch":
            breakpoint()
            x = QuantumModelAutoGrad.apply(
                x,
                *list(self.parameters()),
                self.layers,
                self.backend,
                self.differentiation,
            )
        else:
            x = _run_layers(x, self.layers)
        return x

    @property
    def nqubits(self) -> int:
        return self.layers[0].circuit.nqubits

    @property
    def backend(self) -> Backend:
        return self.layers[0].backend

    @property
    def output_shape(self):
        return self.layers[-1].output_shape


class QuantumModelAutoGrad(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        *parameters,
        layers: list[QuantumCircuitLayer],
        backend,
        differentiation,
    ):
        ctx.save_for_backward(x)
        ctx.layers = layers
        ctx.differentiation = differentiation
        x_clone = x.clone().detach().numpy()
        x_clone = backend.cast(x_clone, dtype=x_clone.dtype)
        x_clone = torch.as_tensor(np.array(_run_layers(x_clone, layers)))
        x_clone.requires_grad = True
        return x_clone

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (x,) = ctx.saved_tensors
        gradients = ctx.differentiation.evaluate(x, ctx.layers)
        return *gradients, None, None, None
