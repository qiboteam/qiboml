import inspect
import random

import numpy as np
import pytest
from qibo import callbacks, hamiltonians
from qibo.config import raise_error
from qibo.symbols import Z

import qiboml.models.ansatze as ans
import qiboml.models.decoding as dec
import qiboml.models.encoding as enc
from qiboml.operations.differentiation import PSR


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
            else frontend.torch.randn(shape)
        )
    elif (
        frontend.__name__ == "qiboml.interfaces.keras"
        and frontend.keras.backend.backend() == "tensorflow"
    ):
        tensor = (
            frontend.tf.random.uniform(
                shape, minval=0, maxval=2, dtype=frontend.tf.int64
            )
            if binary
            else frontend.tf.random.normal(shape)
        )
    else:
        raise_error(RuntimeError, f"Unknown frontend {frontend}.")
    return tensor


def train_model(frontend, model, data, target):
    max_epochs = 10
    if frontend.__name__ == "qiboml.interfaces.pytorch":

        optimizer = frontend.torch.optim.Adam(model.parameters())
        loss_f = frontend.torch.nn.MSELoss()

        avg_grad, ep = 1.0, 0
        while ep < max_epochs:
            ep += 1
            avg_grad = 0.0
            avg_loss = 0.0
            permutation = frontend.torch.randperm(len(data))
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

    elif frontend.__name__ == "qiboml.interfaces.keras":
        optimizer = frontend.keras.optimizers.Adam()
        loss_f = frontend.keras.losses.MeanSquaredError()

        @frontend.tf.function
        def train_step(x, y):
            with frontend.tf.GradientTape() as tape:
                predictions = model(x)
                loss = loss_f(y, predictions)

            gradients = tape.gradient(
                loss, model.trainable_variables
            )  # Compute gradients
            optimizer.apply_gradients(
                zip(gradients, model.trainable_variables)
            )  # Apply gradients
            return gradients

        def get_avg_grad():
            avg_grad = 0.0
            for x, y in zip(data, target):
                tmp = [
                    frontend.tf.norm(grad)
                    for grad in train_step(x[None, :], y[None, :])
                ]
                avg_grad += sum(tmp) / len(tmp)
            return avg_grad / len(data)

        class GradientStopCallback(frontend.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                grad = get_avg_grad()
                logs["grad"] = grad
                if grad < 1e-2:
                    self.stopped_epoch = epoch
                    self.model.stop_training = True

        run_eagerly = (
            model.decoding.backend.platform != "tensorflow"
            or not model.decoding.analytic
        )

        model.compile(loss=loss_f, optimizer=optimizer, run_eagerly=run_eagerly)
        history = model.fit(
            data,
            target,
            batch_size=1,
            epochs=max_epochs,
            callbacks=[GradientStopCallback()],
        )
        return history.history["grad"][-1]


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
        model.run_eagerly = (
            model.decoding.backend.platform != "tensorflow"
            or not model.decoding.analytic
        )
        for x in data:
            x = frontend.tf.expand_dims(x, axis=0)
            outputs.append(model(x))
        outputs = frontend.tf.stack(outputs, axis=0)
    if target is not None:
        loss = loss_f(target, outputs)
    return outputs, loss


def set_seed(frontend, seed):
    random.seed(seed)
    np.random.seed(seed)
    if frontend.__name__ == "qiboml.interfaces.pytorch":
        frontend.torch.set_default_dtype(frontend.torch.float64)
        frontend.torch.manual_seed(seed)
    elif frontend.__name__ == "qiboml.interfaces.keras":
        frontend.keras.utils.set_random_seed(seed)
        frontend.tf.config.experimental.enable_op_determinism()


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
                val = frontend.tf.random.uniform(weight.shape)
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


@pytest.mark.parametrize("layer,seed", zip(ENCODING_LAYERS, [4, 1]))
def test_encoding(backend, frontend, layer, seed):
    if (
        frontend.__name__ == "qiboml.interfaces.keras"
        and backend.platform != "tensorflow"
    ):
        # pytest.skip("keras interface not ready.")
        pass

    set_device(frontend)
    set_seed(frontend, seed)

    nqubits = 2
    dim = 2

    training_layer = ans.ReuploadingCircuit(
        nqubits,
        random_subset(nqubits, dim),
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

    encoding_layer = layer(nqubits, random_subset(nqubits, dim))
    binary = True if encoding_layer.__class__.__name__ == "BinaryEncoding" else False
    activation = build_activation(frontend, binary)
    q_model = build_sequential_model(
        frontend,
        [
            activation,
            frontend.QuantumModel(
                encoding=encoding_layer,
                circuit=training_layer,
                decoding=decoding_layer,
            ),
        ],
    )
    setattr(q_model, "decoding", decoding_layer)

    data = random_tensor(frontend, (100, dim), binary)
    target = prepare_targets(frontend, q_model, data)

    backprop_test(frontend, q_model, data, target)

    data = random_tensor(frontend, (100, 4))
    model = build_sequential_model(
        frontend,
        [
            build_linear_layer(frontend, 4, dim),
            q_model,
            build_linear_layer(frontend, 1, 1),
        ],
    )
    setattr(model, "decoding", decoding_layer)

    target = prepare_targets(frontend, model, data)

    backprop_test(frontend, model, data, target)


@pytest.mark.parametrize("layer,seed", zip(DECODING_LAYERS, [1, 3, 1, 26]))
def test_decoding(backend, frontend, layer, seed):
    if frontend.__name__ == "qiboml.interfaces.keras" and layer.__name__ == "Samples":
        pytest.skip("keras interface not ready.")
    if not layer.analytic and not layer is dec.Expectation:
        pytest.skip(
            "Expectation layer is the only differentiable decoding when the diffrule is not analytical."
        )

    set_device(frontend)
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
                encoding=encoding_layer,
                circuit=training_layer,
                decoding=decoding_layer,
            ),
        ],
    )

    data = random_tensor(frontend, (100, dim))

    target = prepare_targets(frontend, q_model, data)

    if layer is dec.Samples:
        with pytest.raises(NotImplementedError):
            _ = backprop_test(frontend, q_model, data, target)
        pytest.skip("Skipping the rest of the test for Samples decoding.")

    backprop_test(frontend, q_model, data, target)

    model = build_sequential_model(
        frontend,
        [
            build_linear_layer(frontend, 4, dim),
            q_model,
            build_linear_layer(frontend, decoding_layer.output_shape[-1], 1),
        ],
    )

    data = random_tensor(frontend, (100, 4))
    target = prepare_targets(frontend, model, data)
    backprop_test(frontend, model, data, target)
