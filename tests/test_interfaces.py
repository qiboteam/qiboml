import inspect
import os
import random

import numpy as np
import pytest
from qibo import construct_backend, gates, hamiltonians
from qibo.config import raise_error
from qibo.noise import NoiseModel, PauliError
from qibo.symbols import Z
from qibo.transpiler import NativeGates, Passes, Unroller

import qiboml.models.ansatze as ans
import qiboml.models.decoding as dec
import qiboml.models.encoding as enc

from .utils import set_seed


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
    if frontend.__name__ == "qiboml.interfaces.pytorch":
        return frontend.torch.nn.Linear(input_dim, output_dim)
    elif frontend.__name__ == "qiboml.interfaces.keras":
        layer = frontend.keras.layers.Dense(output_dim)
        layer.build((input_dim,))
        return layer
    else:
        raise_error(RuntimeError, f"Unknown frontend {frontend}.")


def build_sequential_model(frontend, layers):
    if frontend.__name__ == "qiboml.interfaces.pytorch":
        return frontend.torch.nn.Sequential(*layers)
    elif frontend.__name__ == "qiboml.interfaces.keras":
        return frontend.keras.Sequential(layers)
    else:
        raise_error(RuntimeError, f"Unknown frontend {frontend}.")


def build_activation(frontend, binary=False):
    if frontend.__name__ == "qiboml.interfaces.pytorch":

        class Activation(frontend.torch.nn.Module):
            def forward(self, x):
                if not binary:
                    # normalize
                    x = x / x.max()
                    # apply the tanh and rescale by pi
                    return np.pi * frontend.torch.nn.functional.tanh(x)
                return x

    elif frontend.__name__ == "qiboml.interfaces.keras":

        class Activation(frontend.keras.layers.Layer):
            def call(self, x):
                if not binary:
                    # normalize
                    x = x / x.max()
                    # apply the tanh and rescale by pi
                    return np.pi * frontend.keras.activations.tanh(x)
                return x

    else:
        raise_error(RuntimeError, f"Unknown frontend {frontend}.")

    activation = Activation()
    return activation


def random_tensor(frontend, shape, binary=False):
    if (
        frontend.__name__ == "qiboml.interfaces.pytorch"
        or frontend.keras.backend.backend() == "pytorch"
    ):
        tensor = (
            frontend.torch.randint(0, 2, shape).double()
            if binary
            else frontend.torch.randn(shape).double()
        )
    elif (
        frontend.__name__ == "qiboml.interfaces.keras"
        and frontend.keras.backend.backend() == "tensorflow"
    ):
        tensor = (
            frontend.tf.random.uniform(
                shape, minval=0, maxval=2, dtype=frontend.tf.float64
            )
            if binary
            else frontend.tf.random.normal(shape, dtype=frontend.tf.float64)
        )
    else:
        raise_error(RuntimeError, f"Unknown frontend {frontend}.")
    return tensor


def train_model(frontend, model, data, target, max_epochs=5):
    if frontend.__name__ == "qiboml.interfaces.pytorch":

        optimizer = frontend.torch.optim.Adam(model.parameters(), lr=0.1)
        loss_f = frontend.torch.nn.MSELoss()

        avg_grad, ep = 1.0, 0
        while ep < max_epochs:
            ep += 1
            avg_grad = 0.0
            avg_loss = 0.0
            permutation = frontend.torch.randperm(len(data))
            x_data, y_data = (
                (data[permutation], target[permutation])
                if data[0] != None
                else (data, target)
            )
            for x, y in zip(x_data, y_data):
                optimizer.zero_grad()
                if x is not None:
                    loss = loss_f(model(x), y)
                else:
                    loss = model()
                loss.backward()
                avg_grad += list(model.parameters())[-1].grad.norm()
                avg_loss += loss
                optimizer.step()
            avg_grad /= len(data)
            print(f"avg grad: {avg_grad}, avg loss: {avg_loss/len(data)}")
            if avg_grad < 1e-2:
                break

        return avg_grad / len(data)

    elif frontend.__name__ == "qiboml.interfaces.keras":
        optimizer = frontend.keras.optimizers.Adam(learning_rate=0.1)
        loss_f = frontend.keras.losses.MeanSquaredError()

        def train_step(x, y):
            with frontend.tf.GradientTape() as tape:
                if x is not None:
                    predictions = model(x)
                    loss = loss_f(y, predictions)
                else:
                    loss = model()

            gradients = tape.gradient(
                loss, model.trainable_variables
            )  # Compute gradients
            return gradients

        def get_avg_grad():
            avg_grad = 0.0
            for x, y in zip(data, target):
                tmp = [frontend.tf.norm(grad) for grad in train_step(x[None, :], y)]
                avg_grad += sum(tmp) / len(tmp)
            return avg_grad / len(data)

        class GradientStopCallback(frontend.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                grad = get_avg_grad()
                logs["grad"] = grad
                if grad < 1e-2:
                    self.stopped_epoch = epoch
                    self.model.stop_training = True

        if data[0] != None:
            model.compile(loss=loss_f, optimizer=optimizer)
            history = model.fit(
                data,
                target,
                batch_size=1,
                epochs=max_epochs,
                callbacks=[GradientStopCallback()],
            )
            return history.history["grad"][-1]
        else:
            for e in range(max_epochs):
                for _ in range(len(data)):
                    grads = train_step(None, None)
                    optimizer.apply_gradients(zip(grads, model.trainable_variables))
                avg_grad = frontend.tf.norm(grads)
                print(f"-> Loss: {float(model())}, Grad: {avg_grad}")
                if avg_grad < 1e-2:
                    break
            return avg_grad


def eval_model(frontend, model, data, target=None):
    loss = None
    outputs = []

    if frontend.__name__ == "qiboml.interfaces.pytorch":
        loss_f = frontend.torch.nn.MSELoss()
        with frontend.torch.no_grad():
            for x in data:
                outputs.append(model(x))
            shape = model(data[0]).shape
        outputs = frontend.torch.vstack(outputs).reshape((data.shape[0],) + shape)

    elif frontend.__name__ == "qiboml.interfaces.keras":
        loss_f = frontend.keras.losses.MeanSquaredError(
            reduction="sum_over_batch_size",
        )
        for x in data:
            if x is not None:
                x = frontend.tf.expand_dims(x, axis=0)
            outputs.append(model(x))
        outputs = frontend.tf.stack(outputs, axis=0)
    if target is not None:
        loss = loss_f(target, outputs)
    return outputs, loss


def random_parameters(frontend, model):
    if frontend.__name__ == "qiboml.interfaces.pytorch":
        new_params = {}
        for k, v in model.state_dict().items():
            new_params.update(
                {k: v + frontend.torch.randn(v.shape) / 5}
            )  # perturbation of max +- 0.2
            # of the original parameters
    elif frontend.__name__ == "qiboml.interfaces.keras":
        new_params = []
        for weight in model.get_weights():
            if frontend.keras.backend.backend() == "tensorflow":
                val = frontend.tf.random.normal(weight.shape)
            elif frontend.keras.backend.backend() == "pytorch":
                val = frontend.torch.randn(weight.shape)
            elif frontend.keras.backend.backend() == "jax":
                raise NotImplementedError
            new_params += [weight + val / 5]
    return new_params


def get_parameters(frontend, model):
    if frontend.__name__ == "qiboml.interfaces.pytorch":
        return {k: v.clone() for k, v in model.state_dict().items()}
    elif frontend.__name__ == "qiboml.interfaces.keras":
        return model.get_weights()


def set_device(frontend):
    if frontend.__name__ == "qiboml.interfaces.pytorch":
        frontend.torch.set_default_device(
            "cuda:0" if frontend.torch.cuda.is_available() else "cpu"
        )
    elif frontend.__name__ == "qiboml.interfaces.keras":
        # tf should automatically use GPU by default when available
        pass


def set_parameters(frontend, model, params):
    if frontend.__name__ == "qiboml.interfaces.pytorch":
        model.load_state_dict(params)
    elif frontend.__name__ == "qiboml.interfaces.keras":
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
    assert grad < 1e-2
    assert round(float(loss_untrained), 6) >= round(float(loss_trained), 6)
    # in some (unpredictable) cases the gradient and loss
    # start so small that the model doesn't do anything
    # fixing the seed doesn't fix this on all the platforms
    # thus for now I am just allowing the == to cover those
    # specific (rare) cases


@pytest.mark.parametrize("layer,seed", zip(ENCODING_LAYERS, [6, 4]))
def test_encoding(backend, frontend, layer, seed):
    set_device(frontend)
    set_seed(frontend, seed)

    nqubits = 2
    dim = 2
    density_matrix = False

    training_layer = ans.HardwareEfficient(
        nqubits, random_subset(nqubits, dim), density_matrix=density_matrix
    )

    decoding_qubits = random_subset(nqubits, dim)
    observable = hamiltonians.SymbolicHamiltonian(
        sum([Z(int(i)) for i in decoding_qubits]),
        nqubits=nqubits,
        backend=backend,
    )
    decoding_layer = dec.Expectation(
        nqubits=nqubits,
        qubits=decoding_qubits,
        observable=observable,
        backend=backend,
    )

    encoding_layer = layer(
        nqubits, random_subset(nqubits, dim), density_matrix=density_matrix
    )
    circuit_structure = [encoding_layer, training_layer]

    binary = True if encoding_layer.__class__.__name__ == "BinaryEncoding" else False
    activation = build_activation(frontend, binary)
    q_model = build_sequential_model(
        frontend,
        [
            activation,
            frontend.QuantumModel(
                circuit_structure=circuit_structure,
                decoding=decoding_layer,
            ),
        ],
    )
    setattr(q_model, "decoding", decoding_layer)

    data = random_tensor(frontend, (100, dim), binary)
    target = prepare_targets(frontend, q_model, data)

    backprop_test(frontend, q_model, data, target)


@pytest.mark.parametrize("layer,seed", zip(DECODING_LAYERS, [1, 3, 1, 26]))
def test_decoding(backend, frontend, layer, seed):
    if not layer.analytic and not layer is dec.Expectation:
        pytest.skip(
            "Expectation layer is the only differentiable decoding when the diffrule is not analytical."
        )

    set_device(frontend)
    set_seed(frontend, seed)

    nqubits = 2
    dim = 2
    density_matrix = False
    training_layer = ans.HardwareEfficient(
        nqubits, random_subset(nqubits, dim), density_matrix=density_matrix
    )
    encoding_layer = enc.PhaseEncoding(
        nqubits, random_subset(nqubits, dim), density_matrix=density_matrix
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
        kwargs["nshots"] = None

    if layer is dec.Samples:
        kwargs["nshots"] = 1000

    decoding_layer = layer(nqubits, decoding_qubits, **kwargs)

    activation = build_activation(frontend, binary=False)
    q_model = build_sequential_model(
        frontend,
        [
            activation,
            frontend.QuantumModel(
                circuit_structure=[encoding_layer, training_layer],
                decoding=decoding_layer,
            ),
        ],
    )
    setattr(q_model, "decoding", decoding_layer)

    data = random_tensor(frontend, (100, dim))
    target = prepare_targets(frontend, q_model, data)

    if layer is dec.Samples:
        error = (
            NotImplementedError
            if frontend.__name__ != "qiboml.interfaces.keras"
            else frontend.tf.errors.UnimplementedError
        )
        with pytest.raises(error):
            _ = backprop_test(frontend, q_model, data, target)
        assert q_model(data[0][None, :]).shape == (kwargs["nshots"], nqubits)
    else:
        backprop_test(frontend, q_model, data, target)


def test_composition(backend, frontend):
    set_device(frontend)
    set_seed(frontend, 42)

    nqubits = 2
    encoding_layer = random.choice(ENCODING_LAYERS)(nqubits)
    training_layer = ans.HardwareEfficient(nqubits)
    decoding_layer = random.choice(
        list(set(DECODING_LAYERS) - {dec.Samples, dec.State})
    )(
        nqubits, backend=backend
    )  # make sure it's not Samples

    activation = build_activation(
        frontend, binary=encoding_layer.__class__.__name__ == "BinaryEncoding"
    )
    model = build_sequential_model(
        frontend,
        [
            build_linear_layer(frontend, 1, nqubits),
            activation,
            frontend.QuantumModel(
                circuit_structure=[encoding_layer, training_layer],
                decoding=decoding_layer,
            ),
            build_linear_layer(frontend, decoding_layer.output_shape[-1], 1),
        ],
    )
    setattr(model, "decoding", decoding_layer)

    data = random_tensor(frontend, (50, 1))
    target = prepare_targets(frontend, model, data)
    _, loss_untrained = eval_model(frontend, model, data, target)
    train_model(frontend, model, data, target, max_epochs=5)
    _, loss_trained = eval_model(frontend, model, data, target)
    assert loss_untrained > loss_trained


def test_vqe(backend, frontend):
    seed = 42
    set_device(frontend)
    set_seed(frontend, seed)
    backend.set_seed(42)

    tfim = hamiltonians.TFIM(nqubits=2, backend=backend)

    nqubits = 2
    dim = 2
    training_layer = ans.HardwareEfficient(
        nqubits,
        nlayers=2,
    )
    decoding_layer = dec.Expectation(
        nqubits=nqubits,
        backend=backend,
        observable=tfim,
    )
    circuit_structure = [
        training_layer,
    ]
    q_model = frontend.QuantumModel(
        circuit_structure=circuit_structure,
        decoding=decoding_layer,
    )

    none = np.array(
        5
        * [
            None,
        ]
    )
    grad = train_model(frontend, q_model, none, none, max_epochs=10)
    cost = q_model()
    backend.assert_allclose(float(cost), -2.0, atol=5e-2)


def test_noise(backend, frontend):
    set_device(frontend)
    backend.set_seed(42)
    set_seed(frontend, 42)

    nqubits = 6
    noise = NoiseModel()
    noise.add(PauliError([("X", 0.5)]), gates.CNOT)
    noise.add(PauliError([("Y", 0.2)]), gates.RY)
    noise.add(PauliError([("Z", 0.2)]), gates.RZ)

    encoding_layer = random.choice(ENCODING_LAYERS)(nqubits, density_matrix=True)
    training_layer = ans.HardwareEfficient(nqubits, density_matrix=True)
    noisy_training_layer = noise.apply(training_layer.copy())
    circuit = [encoding_layer, training_layer]
    noisy_circuit = [encoding_layer, noisy_training_layer]
    decoding_layer = random.choice(DECODING_LAYERS)(nqubits, backend=backend)
    activation = build_activation(frontend, binary=False)
    model = build_sequential_model(
        frontend,
        [
            activation,
            frontend.QuantumModel(
                circuit_structure=circuit,
                decoding=decoding_layer,
            ),
        ],
    )
    setattr(model, "decoding", decoding_layer)
    noisy_model = build_sequential_model(
        frontend,
        [
            activation,
            frontend.QuantumModel(
                circuit_structure=noisy_circuit,
                decoding=decoding_layer,
            ),
        ],
    )
    setattr(noisy_model, "decoding", decoding_layer)

    data = random_tensor(frontend, (100, nqubits))
    target = prepare_targets(frontend, model, data)
    train_model(frontend, model, data, target, max_epochs=1)
    _, loss = eval_model(frontend, model, data, target)
    train_model(frontend, noisy_model, data, target, max_epochs=1)
    _, noisy_loss = eval_model(frontend, noisy_model, data, target)
    assert noisy_loss > loss


def test_qibolab(frontend):
    try:
        from qibolab import create_platform
    except ImportError:
        pytest.skip("qibolab not installed.")

    os.environ["QIBOLAB_PLATFORMS"] = "tests/"
    platform = create_platform("emulator")
    backend = construct_backend("qibolab", platform=platform)
    transpiler = Passes(
        connectivity=backend.platform.pairs, passes=[Unroller(NativeGates.default())]
    )

    set_device(frontend)
    set_seed(frontend, 42)

    nqubits = 1
    encoding_layer = enc.PhaseEncoding(nqubits)
    training_layer = ans.HardwareEfficient(nqubits)
    decoding_layer = dec.Expectation(
        nqubits, wire_names=["0"], backend=backend, transpiler=transpiler, nshots=1000
    )

    activation = build_activation(
        frontend, binary=encoding_layer.__class__.__name__ == "BinaryEncoding"
    )
    model = build_sequential_model(
        frontend,
        [
            build_linear_layer(frontend, 1, nqubits),
            activation,
            frontend.QuantumModel(
                circuit_structure=[encoding_layer, training_layer],
                decoding=decoding_layer,
            ),
            build_linear_layer(frontend, decoding_layer.output_shape[-1], 1),
        ],
    )
    setattr(model, "decoding", decoding_layer)

    data = random_tensor(frontend, (10, 1))
    target = prepare_targets(frontend, model, data)
    _, loss_untrained = eval_model(frontend, model, data, target)
    train_model(frontend, model, data, target, max_epochs=1)
    _, loss_trained = eval_model(frontend, model, data, target)
    assert loss_untrained > loss_trained
