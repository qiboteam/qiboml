import inspect
import random

import numpy as np
import pytest
import torch
from qibo import construct_backend, hamiltonians
from qibo.config import raise_error
from qibo.symbols import Z

import qiboml.models.ansatze as ans
import qiboml.models.decoding as dec
import qiboml.models.encoding as enc

torch.set_default_dtype(torch.float64)


def get_layers(module, layer_type=None):
    layers = []
    for _, layer in inspect.getmembers(module, inspect.isclass):
        if layer.__module__ == module.__name__:
            if layer_type is not None:
                if issubclass(layer, layer_type) and not layer is layer_type:
                    pass
                else:
                    continue
            layers.append(layer)
    return layers


ENCODING_LAYERS = get_layers(enc, enc.QuantumEncoding)
DECODING_LAYERS = get_layers(dec, dec.QuantumDecoding)
ANSATZE_LAYERS = get_layers(ans)


def random_subset(nqubits, k):
    return np.random.choice(range(nqubits), size=(k,), replace=False).tolist()


def build_linear_layer(frontend, input_dim, output_dim):
    if frontend.__name__ == "qiboml.models.pytorch":
        return frontend.torch.nn.Linear(input_dim, output_dim)
    elif frontend.__name__ == "qiboml.models.keras":
        return frontend.keras.layers.Dense(output_dim)
    else:
        raise_error(RuntimeError, f"Unknown frontend {frontend}.")


def build_sequential_model(frontend, layers, binary=False):
    if frontend.__name__ == "qiboml.models.pytorch":
        activation = frontend.torch.nn.Threshold(1, 0)
        layers = layers[:1] + [activation] + layers[1:] if binary else layers
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
    max_epochs = 30
    if frontend.__name__ == "qiboml.models.pytorch":

        optimizer = torch.optim.Adam(model.parameters())
        loss_f = torch.nn.MSELoss()

        avg_grad, ep = 1.0, 0
        shape = model(data[0]).shape
        while ep < max_epochs:
            ep += 1
            avg_grad = 0.0
            avg_loss = 0.0
            permutation = frontend.torch.randint(0, len(data), (len(data),))
            for x, y in zip(data[permutation], target[permutation]):
                optimizer.zero_grad()
                loss = loss_f(model(x), y)
                loss.backward()
                avg_grad += list(model.parameters())[-1].grad.norm()
                avg_loss += loss
                optimizer.step()
            avg_grad /= len(data)
            print(f"avg grad: {avg_grad}, avg loss: {avg_loss/len(data)}")
            if avg_grad < 1e-2:
                break

        return avg_grad / len(data)

    elif frontend.__name__ == "qiboml.models.keras":

        optimizer = frontend.keras.optimizers.Adam()
        loss_f = frontend.keras.losses.MeanSquaredError()
        model.compile(loss=loss_f, optimizer=optimizer)
        model.fit(
            data,
            target,
            batch_size=1,
            epochs=epochs,
        )


def eval_model(frontend, model, data, target=None):
    loss = None
    outputs = []

    if frontend.__name__ == "qiboml.models.pytorch":
        loss_f = torch.nn.MSELoss()
        with torch.no_grad():
            for x in data:
                outputs.append(model(x))
            shape = model(data[0]).shape
        outputs = frontend.torch.vstack(outputs).reshape((data.shape[0],) + shape)

    elif frontend.__name__ == "qiboml.models.keras":
        loss_f = frontend.keras.losses.MeanSquaredError(
            reduction="sum_over_batch_size",
        )
        for x in data:
            outputs.append(model(x))
        outputs = frontend.tf.stack(outputs, axis=0)
    if target is not None:
        loss = loss_f(target, outputs)
    return outputs, loss


def set_seed(frontend, seed):
    random.seed(seed)
    np.random.seed(seed)
    if frontend.__name__ == "qiboml.models.pytorch":
        frontend.torch.manual_seed(seed)


def random_parameters(frontend, model):
    if frontend.__name__ == "qiboml.models.pytorch":
        new_params = {}
        for k, v in model.state_dict().items():
            new_params.update({k: v + frontend.torch.randn(v.shape) / 2})
    elif frontend.__name__ == "qiboml.models.keras":
        new_params = [frontend.tf.random.uniform(model.get_weights()[0].shape)]
    return new_params


def get_parameters(frontend, model):
    if frontend.__name__ == "qiboml.models.pytorch":
        return {k: v.clone() for k, v in model.state_dict().items()}
    elif frontend.__name__ == "qiboml.models.keras":
        return model.get_weights()


def set_parameters(frontend, model, params):
    if frontend.__name__ == "qiboml.models.pytorch":
        model.load_state_dict(params)
    elif frontend.__name__ == "qiboml.models.keras":
        model.set_weights(params)


def prepare_targets(frontend, model, data):
    target_params = random_parameters(frontend, model)
    init_params = get_parameters(frontend, model)
    set_parameters(frontend, model, target_params)
    target, _ = eval_model(frontend, model, data)
    set_parameters(frontend, model, init_params)
    return target


def backprop_test(frontend, model, data, target):
    _, loss_untrained = eval_model(frontend, model, data, target)
    grad = train_model(frontend, model, data, target)
    _, loss_trained = eval_model(frontend, model, data, target)
    assert loss_untrained > loss_trained
    assert grad < 1e-2


@pytest.mark.parametrize("layer,seed", zip(ENCODING_LAYERS, [1, 4]))
def test_encoding(backend, frontend, layer, seed):
    if frontend.__name__ == "qiboml.models.keras":
        pytest.skip("keras interface not ready.")
    if backend.name not in ("pytorch", "jax"):
        pytest.skip("Non pytorch/jax differentiation is not working yet.")

    set_seed(frontend, seed)

    nqubits = 2
    dim = 2
    training_layer = ans.ReuploadingCircuit(
        nqubits,
        random_subset(nqubits, dim),
    )
    decoding_layer = dec.Probabilities(
        nqubits, random_subset(nqubits, dim), backend=backend
    )
    encoding_layer = layer(nqubits, random_subset(nqubits, dim))
    q_model = frontend.QuantumModel(encoding_layer, training_layer, decoding_layer)
    binary = True if encoding_layer.__class__.__name__ == "BinaryEncoding" else False
    data = random_tensor(frontend, (100, dim), binary)
    target = prepare_targets(frontend, q_model, data)
    backprop_test(frontend, q_model, data, target)

    data = random_tensor(frontend, (100, 32))
    model = build_sequential_model(
        frontend,
        [
            build_linear_layer(frontend, 32, dim),
            q_model,
            build_linear_layer(frontend, 2**nqubits, 1),
        ],
        binary=binary,
    )
    target = prepare_targets(frontend, model, data)
    backprop_test(frontend, model, data, target)


@pytest.mark.parametrize("layer,seed", zip(DECODING_LAYERS, [1, 2, 1, 1]))
@pytest.mark.parametrize("analytic", [True, False])
def test_decoding(backend, frontend, layer, seed, analytic):
    if frontend.__name__ == "qiboml.models.keras":
        pytest.skip("keras interface not ready.")
    if backend.name not in ("pytorch", "jax"):
        pytest.skip("Non pytorch/jax differentiation is not working yet.")
    if analytic and not layer is dec.Expectation:
        pytest.skip("Unused analytic argument.")

    set_seed(frontend, seed)

    nqubits = 2
    dim = 2
    training_layer = ans.ReuploadingCircuit(
        nqubits,
        random_subset(nqubits, dim),
    )
    encoding_layer = enc.PhaseEncoding(
        nqubits,
        random_subset(nqubits, dim),
    )
    kwargs = {"backend": backend}
    decoding_qubits = random_subset(nqubits, dim)
    if layer is dec.Expectation:
        observable = hamiltonians.SymbolicHamiltonian(
            sum([Z(int(i)) for i in decoding_qubits]),
            nqubits=nqubits,
            backend=backend,
        )
        kwargs["observable"] = observable
        kwargs["analytic"] = analytic
        kwargs["nshots"] = None
    decoding_layer = layer(nqubits, decoding_qubits, **kwargs)

    # if not decoding_layer.analytic:
    #     pytest.skip("PSR differentiation is not working yet.")

    q_model = frontend.QuantumModel(encoding_layer, training_layer, decoding_layer)

    data = random_tensor(frontend, (100, dim))
    target = prepare_targets(frontend, q_model, data)
    backprop_test(frontend, q_model, data, target)

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
    backprop_test(frontend, model, data, target)
