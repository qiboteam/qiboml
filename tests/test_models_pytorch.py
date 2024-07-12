import inspect

import numpy as np
import pytest
import torch
from qibo.config import raise_error

import qiboml.models.ansatze as ans
import qiboml.models.encoding_decoding as ed
from qiboml.models.pytorch import QuantumModel


def get_layers(module, layer_type=None):
    layers = []
    for name, layer in inspect.getmembers(module, inspect.isclass):
        if layer.__module__ == module.__name__:
            if layer_type is not None:
                if issubclass(layer, layer_type):
                    pass
                else:
                    continue
            layers.append(layer)
    return layers


ENCODING_LAYERS = get_layers(ed, ed.QuantumEncodingLayer)
DECODING_LAYERS = get_layers(ed, ed.QuantumDecodingLayer)
ANSATZE_LAYERS = get_layers(ans)


def random_subset(nqubits, k):
    return np.random.choice(range(nqubits), size=(k,), replace=False)


@pytest.mark.parametrize("layer", ENCODING_LAYERS)
def test_pytorch_encoding(layer):
    nqubits = 5
    dim = 4
    training_layer = ans.ReuploadingLayer(nqubits, random_subset(nqubits, dim))
    decoding_layer = ed.ProbabilitiesLayer(nqubits, random_subset(nqubits, dim))
    encoding_layer = layer(nqubits, random_subset(nqubits, dim))
    q_model = QuantumModel(
        layers=[
            encoding_layer,
            training_layer,
            decoding_layer,
        ]
    )
    model = torch.nn.Sequential(
        torch.nn.Linear(128, dim),
        torch.nn.Hardshrink(),
        q_model,
        torch.nn.Linear(2**nqubits, 1),
    )
    data = torch.randn(1, 128)
    model(data)
