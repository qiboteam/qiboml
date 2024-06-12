import inspect

import torch

import qiboml.models.encoding_decoding as encodings
from qiboml.models.abstract import QuantumCircuitLayer


def __init__factory(layer):
    def __init__(cls, *args, **kwargs):
        nonlocal layer
        torch.nn.Module.__init__(cls)
        layer.__init__(cls, *args, **kwargs)
        cls.register_parameter(
            layer.__name__,
            torch.nn.Parameter(torch.as_tensor(cls.circuit.get_parameters())),
        )

    return __init__


for name, layer in inspect.getmembers(encodings, inspect.isclass):
    if layer.__module__ == encodings.__name__:
        newcls = type(
            name,
            (torch.nn.Module, layer),
            {
                "__init__": __init__factory(layer),
            },
        )
        newcls(3)
