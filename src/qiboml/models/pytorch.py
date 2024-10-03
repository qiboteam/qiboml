"""Torch interface to qiboml layers"""

from dataclasses import dataclass
from typing import Generator

import numpy as np
import torch
from qibo import Circuit
from qibo.backends import Backend, _check_backend
from qibo.config import raise_error
from torch.autograd import forward_ad

import qiboml.models.encoding_decoding as ed
from qiboml.models.abstract import QuantumCircuitLayer, _run_layers
from qiboml.models.decoding import QuantumDecoding
from qiboml.models.encoding import QuantumEncoding
from qiboml.operations import differentiation as Diff

BACKEND_2_DIFFERENTIATION = {
    "pytorch": None,
    "tensorflow": "_PSR",
    "jax": "_PSR",
}


@dataclass(eq=False)
class QuantumModel(torch.nn.Module):

    layers: list[QuantumCircuitLayer]
    differentiation: str = "auto"

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

        for j, layer in enumerate(self._trainable_layers):
            for i, params in enumerate(layer.parameters):
                if self.backend.name != "pytorch":
                    params = [p.tolist() for p in params]
                params = torch.as_tensor(params)
                params.requires_grad = True
                setattr(
                    self,
                    f"{layer.__class__.__name__}-{j}_{i}",
                    torch.nn.Parameter(params.squeeze()),
                )

        if self.backend.name == "pytorch":
            for i, layer in enumerate(self._trainable_layers):
                layer.parameters = self._get_parameters_for_layer(i)

        if self.differentiation == "auto":
            self.differentiation = BACKEND_2_DIFFERENTIATION.get(
                self.backend.name, "_PSR"
            )

        if self.differentiation is not None:
            self.differentiation = getattr(Diff, self.differentiation)()

    def forward(self, x: torch.Tensor):
        if self.backend.name != "pytorch" or self.differentiation is not None:
            x = QuantumModelAutoGrad.apply(
                x, self.layers, self.backend, self.differentiation, *self.parameters()
            )
        else:
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

    @property
    def _trainable_layers(
        self,
    ) -> Generator[QuantumCircuitLayer, None, None]:
        return (layer for layer in self.layers if layer.has_parameters)

    def _get_parameters_for_layer(self, i: int) -> list[torch.nn.Parameter]:
        layer_name = list(self._trainable_layers)[i].__class__.__name__
        return [v for k, v in self.named_parameters() if f"{layer_name}-{i}_" in k]


class QuantumModelAutoGrad(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        layers: list[QuantumCircuitLayer],
        backend,
        differentiation,
        *parameters: list[torch.nn.Parameter],
    ):
        ctx.save_for_backward(x, *parameters)
        ctx.layers = layers
        ctx.differentiation = differentiation
        x_clone = x.clone().detach().numpy()
        x_clone = backend.cast(x_clone, dtype=x_clone.dtype)
        params = [
            backend.cast(par.clone().detach().numpy(), dtype=backend.precision)
            for par in parameters
        ]

        index = 0
        for layer in layers:
            if layer.has_parameters:
                nparams = len(list(layer.parameters))
                layer.parameters = params[index : index + nparams]
                index += nparams
            x_clone = layer(x_clone)

        x_clone = torch.as_tensor(x_clone.tolist())
        x_clone.requires_grad = True
        return x_clone

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x, *parameters = ctx.saved_tensors
        gradients = [
            torch.as_tensor(grad.tolist())
            for grad in ctx.differentiation.evaluate(x, ctx.layers, *parameters)
        ]
        return (
            grad_output @ gradients[0].transpose(-1, -2),
            None,
            None,
            None,
            *gradients,
        )


@dataclass(eq=False)
class _QuantumModel(torch.nn.Module):

    encoding: QuantumEncoding
    circuit: Circuit
    decoding: QuantumDecoding
    differentiation: str = "auto"

    def __post_init__(
        self,
    ):
        super().__init__()

        params = [p for param in self.circuit.get_parameters() for p in param]
        params = torch.as_tensor(self.backend.to_numpy(params)).ravel()
        params.requires_grad = True
        self.circuit_parameters = torch.nn.Parameter(params)

        if self.differentiation == "auto":
            self.differentiation = BACKEND_2_DIFFERENTIATION.get(
                self.backend.name, "_PSR"
            )

        if self.differentiation is not None:
            self.differentiation = getattr(Diff, self.differentiation)()

    def forward(self, x: torch.Tensor):
        if self.backend.name != "pytorch" or self.differentiation is not None:
            x = _QuantumModelAutoGrad.apply(
                x,
                self.encoding,
                self.circuit,
                self.decoding,
                self.backend,
                self.differentiation,
                *list(self.parameters())[0],
            )
        else:
            # breakpoint()
            self.circuit.set_parameters(list(self.parameters())[0])
            x = self.encoding(x) + self.circuit
            x = self.decoding(x)

        return x

    @property
    def nqubits(
        self,
    ) -> int:
        return self.encoding.nqubits

    @property
    def backend(
        self,
    ) -> Backend:
        return self.decoding.backend


class _QuantumModelAutoGrad(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        encoding: QuantumEncoding,
        circuit: Circuit,
        decoding: QuantumDecoding,
        backend,
        differentiation,
        *parameters: list[torch.nn.Parameter],
    ):
        ctx.save_for_backward(x, *parameters)
        ctx.encoding = encoding
        ctx.circuit = circuit
        ctx.decoding = decoding
        ctx.backend = backend
        ctx.differentiation = differentiation
        x_clone = x.clone().detach().cpu().numpy()
        x_clone = backend.cast(x_clone, dtype=x_clone.dtype)
        params = [
            backend.cast(par.clone().detach().cpu().numpy(), dtype=backend.precision)
            for par in parameters
        ]
        x_clone = encoding(x_clone) + circuit
        x_clone.set_parameters(params)
        x_clone = decoding(x_clone)
        x_clone = torch.as_tensor(backend.to_numpy(x_clone))
        return x_clone

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x, *parameters = ctx.saved_tensors
        x_clone = x.clone().detach().cpu().numpy()
        x_clone = ctx.backend.cast(x_clone, dtype=x_clone.dtype)
        params = [
            ctx.backend.cast(
                par.clone().detach().cpu().numpy(), dtype=ctx.backend.precision
            )
            for par in parameters
        ]
        gradients = [
            torch.as_tensor(ctx.backend.to_numpy(grad))
            for grad in ctx.differentiation.evaluate(
                x_clone, ctx.encoding, ctx.circuit, ctx.decoding, ctx.backend, *params
            )
        ]
        return (
            grad_output * torch.as_tensor(gradients),
            None,
            None,
            None,
            None,
            None,
            *gradients,
        )
