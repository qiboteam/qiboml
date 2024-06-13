"""Torch interface to qiboml layers"""

import inspect
from dataclasses import dataclass

import torch

import qiboml.models.ansatze as ans
import qiboml.models.encoding_decoding as ed


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

            globals()[name] = dataclass(
                type(
                    name,
                    (torch.nn.Module, layer),
                    {
                        "__init__": __init__,
                        "forward": layer.forward,
                        "backward": layer.backward,
                        "__hash__": torch.nn.Module.__hash__,
                    },
                )
            )


for module in (ed, ans):
    _torch_factory(module)
