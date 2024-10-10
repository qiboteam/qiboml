import inspect

import numpy as np
import pytest
import torch
from qibo import construct_backend, hamiltonians
from qibo.config import raise_error
from qibo.symbols import Z

import qiboml.models.ansatze as ans
import qiboml.models.encoding_decoding as ed

torch.set_default_dtype(torch.float64)


def get_layers(module, layer_type=None):
    layers = []
    for name, layer in inspect.getmembers(module, inspect.isclass):
        if layer.__module__ == module.__name__:
            if layer_type is not None:
                if issubclass(layer, layer_type) and not layer is layer_type:
                    pass
                else:
                    continue
            layers.append(layer)
    return layers


ENCODING_LAYERS = get_layers(ed, ed.QuantumEncodingLayer)
DECODING_LAYERS = get_layers(ed, ed.QuantumDecodingLayer)
ANSATZE_LAYERS = get_layers(ans)


def random_subset(nqubits, k):
    return np.random.choice(range(nqubits), size=(k,), replace=False).tolist()


def build_linear_layer(frontend, input_dim, output_dim):
    import keras

    if frontend.__name__ == "qiboml.models.pytorch":
        return torch.nn.Linear(input_dim, output_dim)
    elif frontend.__name__ == "qiboml.models.keras":
        return keras.layers.Dense(output_dim)
    else:
        raise_error(RuntimeError, f"Unknown frontend {frontend}.")


def build_sequential_model(frontend, layers):
    import keras

    if frontend.__name__ == "qiboml.models.pytorch":
        return torch.nn.Sequential(*layers)
    elif frontend.__name__ == "qiboml.models.keras":
        return keras.Sequential(layers)
    else:
        raise_error(RuntimeError, f"Unknown frontend {frontend}.")


def random_tensor(frontend, shape):
    import tensorflow as tf

    if frontend.__name__ == "qiboml.models.pytorch":
        return torch.randn(shape)
    elif frontend.__name__ == "qiboml.models.keras":
        return tf.random.uniform(shape)
    else:
        raise_error(RuntimeError, f"Unknown frontend {frontend}.")


@pytest.mark.parametrize("layer", ENCODING_LAYERS)
def test_encoding(backend, frontend, layer):
    nqubits = 5
    dim = 4
    training_layer = ans.ReuploadingLayer(
        nqubits, random_subset(nqubits, dim), backend=backend
    )
    decoding_layer = ed.ProbabilitiesLayer(
        nqubits, random_subset(nqubits, dim), backend=backend
    )
    encoding_layer = layer(nqubits, random_subset(nqubits, dim), backend=backend)
    q_model = frontend.QuantumModel(
        layers=[
            encoding_layer,
            training_layer,
            decoding_layer,
        ]
    )
    model = build_sequential_model(
        frontend,
        [
            build_linear_layer(frontend, 128, dim),
            q_model,
            build_linear_layer(frontend, 2**dim, 1),
        ],
    )
    data = random_tensor(frontend, (1, 128))
    model(data)


@pytest.mark.parametrize("layer", DECODING_LAYERS)
@pytest.mark.parametrize("analytic", [True, False])
def test_decoding(backend, frontend, layer, analytic):
    if analytic and not layer is ed.ExpectationLayer:
        pytest.skip("Unused analytic argument.")
    nqubits = 5
    dim = 4
    training_layer = ans.ReuploadingLayer(
        nqubits, random_subset(nqubits, dim), backend=backend
    )
    encoding_layer = ed.PhaseEncodingLayer(
        nqubits, random_subset(nqubits, dim), backend=backend
    )
    kwargs = {"backend": backend}
    decoding_qubits = random_subset(nqubits, dim)
    if layer is ed.ExpectationLayer:
        observable = hamiltonians.SymbolicHamiltonian(
            sum([Z(int(i)) for i in decoding_qubits]),
            nqubits=nqubits,
            backend=backend,
        )
        kwargs["observable"] = observable
        kwargs["analytic"] = analytic
    decoding_layer = layer(nqubits, decoding_qubits, **kwargs)
    q_model = frontend.QuantumModel(
        layers=[
            encoding_layer,
            training_layer,
            decoding_layer,
        ]
    )
    model = build_sequential_model(
        frontend,
        [
            build_linear_layer(frontend, 128, dim),
            q_model,
            build_linear_layer(frontend, q_model.output_shape[-1], 1),
        ],
    )
    data = random_tensor(frontend, (1, 128))
    model(data)


def test_nqubits_error(frontend):
    nqubits = 5
    training_layer = ans.ReuploadingLayer(nqubits - 1)
    decoding_layer = ed.ProbabilitiesLayer(nqubits)
    encoding_layer = ed.BinaryEncodingLayer(nqubits)
    with pytest.raises(RuntimeError):
        frontend.QuantumModel([encoding_layer, training_layer, decoding_layer])


def test_backend_error(frontend):
    numpy = construct_backend("numpy")
    torch = construct_backend("pytorch")
    nqubits = 5
    training_layer = ans.ReuploadingLayer(nqubits, backend=numpy)
    decoding_layer = ed.ProbabilitiesLayer(nqubits, backend=torch)
    encoding_layer = ed.BinaryEncodingLayer(nqubits, backend=numpy)
    with pytest.raises(RuntimeError):
        frontend.QuantumModel([encoding_layer, training_layer, decoding_layer])


def test_final_layer_error(frontend):
    nqubits = 5
    training_layer = ans.ReuploadingLayer(nqubits)
    decoding_layer = ed.ProbabilitiesLayer(nqubits)
    encoding_layer = ed.BinaryEncodingLayer(nqubits)
    with pytest.raises(RuntimeError):
        frontend.QuantumModel([encoding_layer, decoding_layer, training_layer])
