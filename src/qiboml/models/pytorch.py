"""Torch interface to qiboml layers"""

from dataclasses import dataclass

import numpy as np
import torch
from qibo.backends import Backend
from qibo.config import raise_error
from sympy.core import parameters

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
        if not isinstance(self.layers[-1], ed.QuantumDecodingLayer):
            raise_error(
                RuntimeError,
                f"The last layer has to be a `QuantumDecodinglayer`, but is {self.layers[-1]}",
            )

        for layer in self.layers:
            for i, params in enumerate(layer.parameters):
                # if self.backend.name != "pytorch":
                #    params = torch.as_tensor(
                #        params, dtype=self.backend.precision
                #    )
                setattr(
                    self,
                    f"{layer.__class__.__name__}_{i}",
                    torch.nn.Parameter(torch.as_tensor(params)),
                )

        self.differentiation = getattr(Diff, self.differentiation)()

    def forward(self, x: torch.Tensor):
        if self.backend.name != "pytorch":
            x = QuantumModelAutoGrad.apply(
                x, self.layers, self.backend, self.differentiation, *self.parameters()
            )
        else:
            # breakpoint()
            """
            index = 0
            circ = self.layers[0](x)
            parameters = list(self.parameters())
            for layer in self.layers[1:-1]:
                if layer.has_parameters and not issubclass(layer.__class__, ed.QuantumEncodingLayer):
                    layer.parameters = parameters[index]
                    index += 1
                circ = layer.forward(circ)
            #circ.set_parameters([p for param in self.parameters() for p in param])
            return self.layers[-1](circ)
            #x = _run_layers(x, self.layers, list(self.parameters()))
            """
            for layer in self.layers:
                x = layer(x)

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
        layers: list[QuantumCircuitLayer],
        backend,
        differentiation,
        *parameters,
    ):
        ctx.save_for_backward(x, *parameters)
        ctx.layers = layers
        ctx.differentiation = differentiation
        x_clone = x.clone().detach().numpy()
        x_clone = backend.cast(x_clone, dtype=x_clone.dtype)
        x_clone = torch.as_tensor(np.array(_run_layers(x_clone, layers, parameters)))
        x_clone.requires_grad = True
        return x_clone

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (
            x,
            *parameters,
        ) = ctx.saved_tensors
        gradients = [
            torch.as_tensor(grad.tolist(), dtype=torch.float64)
            for grad in ctx.differentiation.evaluate(x, ctx.layers)
        ]
        return (
            grad_output @ gradients[0].transpose(-1, -2),
            None,
            None,
            None,
            *gradients,
        )
