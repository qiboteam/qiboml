"""Torch interface to qiboml layers"""

import inspect
from dataclasses import dataclass

import torch

import qiboml.models.ansatze as ans
import qiboml.models.encoding_decoding as ed
from qiboml.models.abstract import QuantumCircuitLayer


def _torch_factory(module) -> None:
    for name, layer in inspect.getmembers(module, inspect.isclass):
        if layer.__module__ == module.__name__:

            def __init__(cls, *args, **kwargs):
                nonlocal layer
                torch.nn.Module.__init__(cls)
                layer.__init__(cls, *args, **kwargs)
                if len(cls.circuit.get_parameters()) > 0:
                    cls.register_parameter(
                        layer.__name__,
                        torch.nn.Parameter(
                            torch.as_tensor(cls.circuit.get_parameters())
                        ),
                    )

            forward = layer.forward
            if (
                issubclass(layer, ed.QuantumDecodingLayer)
                and layer.__name__ != "QuantumDecodingLayer"
            ):
                forward = lambda *args, **kwargs: torch.as_tensor(
                    forward(*args, **kwargs)
                )

            globals()[name] = dataclass(
                type(
                    name,
                    (torch.nn.Module, layer),
                    {
                        "__init__": __init__,
                        "forward": forward,
                        "backward": layer.backward,
                        "__hash__": torch.nn.Module.__hash__,
                    },
                )
            )


for module in (ed, ans):
    _torch_factory(module)


@dataclass
class QuantumModel(torch.nn.Module):

    def __init__(self, layers: list[QuantumCircuitLayer]):
        super().__init__()
        nqubits = layers[0].circuit.nqubits
        self.layers = layers
        for layer in layers[1:]:
            if layer.circuit.nqubits != nqubits:
                raise_error(
                    RuntimeError,
                    f"Layer \n{layer}\n has {layer.circuit.nqubits} qubits, but {nqubits} qubits was expected.",
                )
        if not isinstance(layers[-1], ed.QuantumDecodingLayer):
            raise_error(
                RuntimeError,
                f"The last layer has to be a `QuantumDecodinglayer`, but is {layers[-1]}",
            )

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, input_grad: torch.Tensor):
        grad = input_grad
        for layer in self.layers:
            grad = layer.backward(grad)
        return grad

    @property
    def nqubits(self):
        return self.layers[0].circuit.nqubits
