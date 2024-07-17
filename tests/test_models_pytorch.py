import inspect

import numpy as np
import pytest
import torch
from qibo import hamiltonians
from qibo.config import raise_error
from qibo.symbols import Z

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


def output_dim(decoding_layer, input_dim, nqubits):
    if decoding_layer is ed.ExpectationLayer:
        return 1
    elif decoding_layer is ed.ProbabilitiesLayer:
        return 2**input_dim
    elif decoding_layer is ed.SamplesLayer:
        return input_dim
    elif decoding_layer is ed.StateLayer:
        return 2**nqubits
    else:
        raise_error(RuntimeError, f"Layer {decoding_layer} not supported.")


@pytest.mark.parametrize("layer", ENCODING_LAYERS)
def test_pytorch_encoding(layer):
    torch.set_default_dtype(torch.float64)
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
        torch.nn.Linear(2**dim, 1),
    )
    data = torch.randn(1, 128)
    model(data)


@pytest.mark.parametrize("layer", DECODING_LAYERS)
def test_pytorch_decoding(layer):
    torch.set_default_dtype(torch.float64)
    nqubits = 5
    dim = 4
    training_layer = ans.ReuploadingLayer(nqubits, random_subset(nqubits, dim))
    encoding_layer = ed.PhaseEncodingLayer(nqubits, random_subset(nqubits, dim))
    kwargs = {}
    if layer is ed.ExpectationLayer:
        observable = hamiltonians.SymbolicHamiltonian(
            sum([Z(int(i)) for i in random_subset(nqubits, dim)])
        )
        kwargs["observable"] = observable
    decoding_layer = layer(nqubits, random_subset(nqubits, dim), **kwargs)
    q_model = QuantumModel(
        layers=[
            encoding_layer,
            training_layer,
            decoding_layer,
        ]
    )
    model = torch.nn.Sequential(
        torch.nn.Linear(128, dim),
        torch.nn.ReLU(),
        q_model,
        torch.nn.Linear(output_dim(layer, dim, nqubits), 128),
    )
    data = torch.randn(1, 128)
    model(data)
