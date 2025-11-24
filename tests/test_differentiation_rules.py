import copy
import random

import numpy as np
import pytest
from qibo import Circuit, gates, hamiltonians
from qibo.backends import NumpyBackend

from qiboml.backends import PyTorchBackend, TensorflowBackend
from qiboml.models.ansatze import hardware_efficient
from qiboml.models.decoding import Expectation
from qiboml.models.encoding import PhaseEncoding
from qiboml.operations.differentiation import PSR, Adjoint, Jax

# TODO: use the classical conftest mechanism or customize mechanism for this test
EXECUTION_BACKENDS = [
    NumpyBackend(),
]
DIFF_RULES = [Jax, PSR, Adjoint]


def set_seed(frontend, seed):
    random.seed(seed)
    np.random.seed(seed)
    frontend.np.random.seed(seed)
    if frontend.__name__ == "qiboml.interfaces.pytorch":
        frontend.torch.set_default_dtype(frontend.torch.float64)
        frontend.torch.manual_seed(seed)
    elif frontend.__name__ == "qiboml.interfaces.keras":
        frontend.keras.backend.set_floatx("float64")
        frontend.tf.keras.backend.set_floatx("float64")
        frontend.keras.utils.set_random_seed(seed)
        frontend.tf.config.experimental.enable_op_determinism()


def construct_x(frontend, wrt_inputs=False):
    if frontend.__name__ == "qiboml.interfaces.pytorch":
        x = frontend.torch.tensor([0.5, 0.8], requires_grad=wrt_inputs)
        return x
    elif frontend.__name__ == "qiboml.interfaces.keras":
        if frontend.keras.backend.backend() == "tensorflow":
            x = (
                frontend.tf.Variable([0.5, 0.8])
                if wrt_inputs
                else frontend.tf.constant([0.5, 0.8])
            )
            return x
        elif frontend.keras.backend.backend() == "pytorch":
            raise NotImplementedError
        else:
            raise NotImplementedError


def compute_gradient(frontend, model, x, wrt_inputs=False):
    if frontend.__name__ == "qiboml.interfaces.keras":
        if frontend.keras.backend.backend() == "tensorflow":
            with frontend.tf.GradientTape(persistent=True) as tape:
                xx = 2.0 * x if wrt_inputs else x
                if wrt_inputs:
                    tape.watch(xx)
                model.circuit_parameters = frontend.tf.Variable(
                    model.circuit_parameters
                )
                expval = model(xx)
            grad_wrt_input = tape.gradient(expval, x)
            grad_wrt_params = tape.gradient(expval, model.circuit_parameters)
            return grad_wrt_input, grad_wrt_params
        elif frontend.keras.backend.backend() == "pytorch":
            raise NotImplementedError
        else:
            raise NotImplementedError

    elif frontend.__name__ == "qiboml.interfaces.pytorch":
        xx = 2.0 * x if wrt_inputs else x
        expval = model(xx)
        expval.backward()
        grad_wrt_input = np.array(x.grad)
        grad_wrt_params = np.array(list(model.parameters())[-1].grad)
        return grad_wrt_input, grad_wrt_params


def gradient_test_setup(
    circuit,
    x,
    nqubits,
    nshots,
    frontend,
    backend,
    diff_rule,
    wrt_inputs,
    initial_params,
):
    decoding_layer = Expectation(
        nqubits=nqubits,
        backend=backend,
        nshots=nshots,
    )

    q_model = frontend.QuantumModel(
        circuit_structure=circuit,
        decoding=decoding_layer,
        differentiation=diff_rule,
        parameters_initialization=initial_params,
    )

    grad_wrt_input, grad_wrt_params = compute_gradient(
        frontend, q_model, x, wrt_inputs=wrt_inputs
    )
    return np.array(grad_wrt_input), np.array(grad_wrt_params)


@pytest.mark.parametrize("nshots", [None, 12000000])
@pytest.mark.parametrize("backend", EXECUTION_BACKENDS)
@pytest.mark.parametrize("diff_rule", DIFF_RULES)
@pytest.mark.parametrize("wrt_inputs", [True, False])
def test_expval_custom_grad(
    frontend,
    backend,
    nshots,
    diff_rule,
    wrt_inputs,
):

    if diff_rule is not None and diff_rule.__name__ == "Jax" and nshots is not None:
        pytest.skip("Jax differentiation does not work with shots.")

    native_backend = (
        PyTorchBackend()
        if frontend.__name__ == "qiboml.interfaces.pytorch"
        else TensorflowBackend()
    )
    seed = 42
    set_seed(frontend, seed)
    backend.set_seed(seed)
    native_backend.set_seed(seed)

    x = construct_x(frontend, wrt_inputs=wrt_inputs)
    x_native = construct_x(frontend, wrt_inputs=wrt_inputs)

    nqubits = 2

    encoding_layer = PhaseEncoding(nqubits=nqubits)
    training_layer_1 = hardware_efficient(nqubits=nqubits, nlayers=1)
    params = np.arange(len(training_layer_1.get_parameters())) * np.pi / 8
    training_layer_1.set_parameters(params)
    training_layer_2 = training_layer_1.copy(deep=True)

    engine = (
        frontend.torch
        if frontend.__name__ == "qiboml.interfaces.pytorch"
        else frontend.keras.ops
    )

    def equivariant_circuit(th, phi, lam):
        c = Circuit(nqubits)
        delta = 2 * engine.cos(phi) + lam**2
        gamma = lam * engine.exp(th / 2)
        c.add([gates.RZ(i, theta=th) for i in range(nqubits)])
        c.add([gates.RX(i, theta=lam) for i in range(nqubits)])
        c.add([gates.RY(i, theta=phi) for i in range(nqubits)])
        c.add(gates.RZ(0, theta=delta))
        c.add(gates.RX(1, theta=gamma))
        return c

    circuit_structure = [
        encoding_layer,
        training_layer_1,
        encoding_layer,
        training_layer_2,
        equivariant_circuit,
    ]

    nparams = 11
    initial_params = np.pi * np.random.randn(nparams)
    initial_params = (
        frontend.torch.as_tensor(initial_params, dtype=frontend.torch.float64)
        if frontend.__name__ == "qiboml.interfaces.pytorch"
        else frontend.keras.ops.cast(initial_params, dtype="float64")
    )

    circuit_structure_native = [copy.deepcopy(c) for c in circuit_structure]

    custom_gradients = gradient_test_setup(
        circuit_structure,
        x,
        nqubits,
        nshots,
        frontend,
        backend,
        diff_rule,
        wrt_inputs,
        initial_params=initial_params,
    )
    native_gradients = gradient_test_setup(
        circuit_structure_native,
        x_native,
        nqubits,
        None,
        frontend,
        native_backend,
        None,
        wrt_inputs,
        initial_params=initial_params,
    )
    tol = 1e-3 if nshots is None else 1e-1
    if wrt_inputs:
        backend.assert_allclose(custom_gradients[0], native_gradients[0], atol=tol)
    else:
        assert custom_gradients[0].item() is None and native_gradients[0].item() is None
    backend.assert_allclose(custom_gradients[1], native_gradients[1], atol=tol)
