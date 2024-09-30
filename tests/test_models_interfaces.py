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
    if frontend.__name__ == "qiboml.models.pytorch":
        return frontend.torch.nn.Sequential(*layers)
    elif frontend.__name__ == "qiboml.models.keras":
        return frontend.keras.Sequential(layers)
    else:
        raise_error(RuntimeError, f"Unknown frontend {frontend}.")


def random_tensor(frontend, shape, binary=False):
    if frontend.__name__ == "qiboml.models.pytorch":
        tensor = frontend.torch.randint(0, 2, shape) if binary else torch.randn(shape)
    elif frontend.__name__ == "qiboml.models.keras":
        tensor = frontend.tf.random.uniform(shape)
    else:
        raise_error(RuntimeError, f"Unknown frontend {frontend}.")
    return tensor


def train_model(frontend, model, data, target):
    if frontend.__name__ == "qiboml.models.pytorch":

        optimizer = torch.optim.LBFGS(model.parameters(), lr=1e-1, tolerance_grad=1e-4)
        loss_f = torch.nn.MSELoss()

        def closure():
            optimizer.zero_grad()
            out = frontend.torch.vstack([model(x) for x in data])
            loss = loss_f(out, target)
            loss.backward()
            return loss

        optimizer.step(closure)
    elif frontend.__name__ == "qiboml.models.keras":
        pass


def eval_model(frontend, model, data, target=None):
    loss = None
    outputs = []
    if frontend.__name__ == "qiboml.models.pytorch":
        loss_f = torch.nn.MSELoss()
        with torch.no_grad():
            for x in data:
                outputs.append(model(x))
            outputs = frontend.torch.vstack(outputs)
    elif frontend.__name__ == "qiboml.models.keras":
        pass
    if target is not None:
        loss = loss_f(target, outputs)
    return outputs, loss


def random_parameters(frontend, model):
    if frontend.__name__ == "qiboml.models.pytorch":
        new_params = {}
        for k, v in model.state_dict().items():
            new_params.update({k: frontend.torch.randn(v.shape)})
    elif frontend.__name__ == "qiboml.models.keras":
        pass
    return new_params


def get_parameters(frontend, model):
    if frontend.__name__ == "qiboml.models.pytorch":
        return {k: v.clone() for k, v in model.state_dict().items()}
    elif frontend.__name__ == "qiboml.models.keras":
        pass


def set_parameters(frontend, model, params):
    if frontend.__name__ == "qiboml.models.pytorch":
        model.load_state_dict(params)
    elif frontend.__name__ == "qiboml.models.keras":
        pass


def prepare_targets(frontend, model, data):
    target_params = random_parameters(frontend, model)
    init_params = get_parameters(frontend, model)
    set_parameters(frontend, model, target_params)
    target, _ = eval_model(frontend, model, data)
    set_parameters(frontend, model, init_params)
    return target


def test_backprop(frontend, model, data, target):
    _, loss_untrained = eval_model(frontend, model, data, target)
    train_model(frontend, model, data, target)
    _, loss_trained = eval_model(frontend, model, data, target)
    assert loss_untrained > loss_trained


@pytest.mark.parametrize("layer", ENCODING_LAYERS)
def test_encoding(backend, frontend, layer):
    nqubits = 3
    dim = 2
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

    binary = True if layer.__class__.__name__ == "BinaryEncodingLayer" else False
    data = random_tensor(frontend, (100, dim), binary)
    target = prepare_targets(frontend, q_model, data)
    test_backprop(frontend, q_model, data, target)

    data = random_tensor(frontend, (100, 32))
    model = build_sequential_model(
        frontend,
        [
            build_linear_layer(frontend, 32, dim),
            q_model,
            build_linear_layer(frontend, 2**dim, 1),
        ],
    )
    target = prepare_targets(frontend, model, data)
    test_backprop(frontend, model, data, target)


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

    data = random_tensor(frontend, (100, dim))
    target = prepare_targets(frontend, q_model, data)
    test_backprop(frontend, q_model, data, target)

    model = build_sequential_model(
        frontend,
        [
            build_linear_layer(frontend, 32, dim),
            q_model,
            build_linear_layer(frontend, q_model.output_shape[-1], 1),
        ],
    )

    data = random_tensor(frontend, (100, 32))
    target = prepare_targets(frontend, model, data)
    test_backprop(frontend, model, data, target)


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
