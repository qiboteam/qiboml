import inspect

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

        optimizer = torch.optim.LBFGS(model.parameters(), lr=1.0, tolerance_grad=1e-3)
        loss_f = torch.nn.MSELoss()

        def closure():
            optimizer.zero_grad()
            out = frontend.torch.vstack([model(x) for x in data])
            loss = loss_f(out, target)
            loss.backward()
            return loss

        optimizer.step(closure)

    elif frontend.__name__ == "qiboml.models.keras":

        optimizer = frontend.keras.optimizers.Adam()
        loss_f = frontend.keras.losses.MeanSquaredError()
        model.compile(loss=loss_f, optimizer=optimizer)
        model.fit(
            data,
            target,
            batch_size=1,
            epochs=500,
        )


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
        loss_f = frontend.keras.losses.MeanSquaredError(
            reduction="sum_over_batch_size",
        )
        for x in data:
            outputs.append(model(x))
        outputs = frontend.tf.stack(outputs, axis=0)
    if target is not None:
        loss = loss_f(target, outputs)
    return outputs, loss


def random_parameters(frontend, model):
    if frontend.__name__ == "qiboml.models.pytorch":
        new_params = {}
        for k, v in model.state_dict().items():
            new_params.update({k: frontend.torch.randn(v.shape)})
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


def assert_grad_norm(frontend, parameters, tol=1e-3):
    if frontend.__name__ == "qiboml.models.pytorch":
        for param in parameters:
            assert param.grad.norm() < tol
    elif frontend.__name__ == "qiboml.models.keras":
        pass


def backprop_test(frontend, model, data, target):
    _, loss_untrained = eval_model(frontend, model, data, target)
    train_model(frontend, model, data, target)
    _, loss_trained = eval_model(frontend, model, data, target)
    assert loss_untrained > loss_trained
    # assert_grad_norm(frontend, model.parameters())


@pytest.mark.parametrize("layer", ENCODING_LAYERS)
def test_encoding(backend, frontend, layer):
    if frontend.__name__ == "qiboml.models.keras":
        pytest.skip("keras interface not ready.")
    if backend.name != "pytorch":
        pytest.skip("Non pytorch differentiatio is not working yet.")
    nqubits = 3
    dim = 2
    training_layer = ans.ReuploadingCircuit(
        nqubits,
        random_subset(nqubits, dim),
    )
    decoding_layer = dec.Probabilities(
        nqubits, random_subset(nqubits, dim), backend=backend
    )
    encoding_layer = layer(nqubits, random_subset(nqubits, dim))
    q_model = frontend.QuantumModel(
        encoding_layer,
        training_layer,
        decoding_layer,
    )

    binary = True if layer.__class__.__name__ == "BinaryEncoding" else False
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
    )
    target = prepare_targets(frontend, model, data)
    backprop_test(frontend, model, data, target)


@pytest.mark.parametrize("layer", DECODING_LAYERS)
@pytest.mark.parametrize("analytic", [True, False])
def test_decoding(backend, frontend, layer, analytic):
    if frontend.__name__ == "qiboml.models.keras":
        pytest.skip("keras interface not ready.")
    if backend.name != "pytorch":
        pytest.skip("Non pytorch differentiatio is not working yet.")
    if analytic and not layer is dec.Expectation:
        pytest.skip("Unused analytic argument.")
    nqubits = 3
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
    decoding_layer = layer(nqubits, decoding_qubits, **kwargs)

    q_model = frontend.QuantumModel(
        encoding_layer, training_layer, decoding_layer, differentiation="Jax"
    )

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
