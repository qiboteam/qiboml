"""Torch interface to qiboml layers"""

from dataclasses import dataclass

import numpy as np
import torch
from qibo.config import raise_error

import qiboml.models.ansatze as ans
import qiboml.models.encoding_decoding as ed
from qiboml.models.abstract import QuantumCircuitLayer


@dataclass
class QuantumModel(torch.nn.Module):

    def __init__(self, layers: list[QuantumCircuitLayer]):
        super().__init__()
        self.layers = layers
        for layer in layers[1:]:
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
        for layer in layers:
            if len(layer.circuit.get_parameters()) > 0:
                self.register_parameter(
                    layer.__class__.__name__,
                    torch.nn.Parameter(torch.as_tensor(layer.circuit.get_parameters())),
                )
        if not isinstance(layers[-1], ed.QuantumDecodingLayer):
            raise_error(
                RuntimeError,
                f"The last layer has to be a `QuantumDecodinglayer`, but is {layers[-1]}",
            )

    def forward(self, x: torch.Tensor):
        if self.backend.name != "pytorch":
            x = x.detach().numpy()
            x = self.backend.cast(x, dtype=x.dtype)
        for layer in self.layers:
            x = layer.forward(x)
        if self.backend.name != "pytorch":
            x = torch.as_tensor(np.array(x))
        return x

    @property
    def nqubits(self) -> int:
        return self.layers[0].circuit.nqubits

    @property
    def backend(self) -> "Backend":
        return self.layers[0].backend

    @property
    def output_shape(self):
        return self.layers[-1].output_shape
