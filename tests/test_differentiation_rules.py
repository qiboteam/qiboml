import random

import numpy as np
import pytest
from qibo import Circuit, gates, hamiltonians
from qibo.backends import NumpyBackend

from qiboml.models.ansatze import hardware_efficient
from qiboml.models.decoding import Expectation
from qiboml.models.encoding import PhaseEncoding
from qiboml.operations.differentiation import PSR, Adjoint, Jax

# TODO: use the classical conftest mechanism or customize mechanism for this test
EXECUTION_BACKENDS = [
    NumpyBackend(),
]
DIFF_RULES = [Jax, PSR, Adjoint]

TARGET_GRAD_TORCH = {
    "no_inputs": (
        None,
        np.array(np.array([5.859873e-01, -1.489167e-04, 1.104562e00, -5.458333e-05])),
    ),
    "wrt_inputs": (
        np.array([1.48141959, -1.46174465]),
        np.array([0.740709794, 0.0, -0.730872325, 0.0]),
    ),
    "equivariant_no_inputs": (
        None,
        np.array(
            [0.458944, -0.524178, 1.215674, -0.241529, -0.723649, 1.270768, 0.162932]
        ),
    ),
    "equivariant_wrt_inputs": (
        np.array([0.911852, -1.561822]),
        np.array(
            [
                0.455926,
                0.387572,
                -0.780911,
                -0.046471,
                0.401802,
                -0.474724,
                0.323378,
            ]
        ),
    ),
}
TARGET_GRAD_TENSORFLOW = TARGET_GRAD_TORCH.copy()
TARGET_GRAD_TENSORFLOW["equivariant_no_inputs"] = (
    None,
    np.array([0.090865, 0.119032, -1.786733, -0.053682, 0.141452, -0.043953, 0.389409]),
)
TARGET_GRAD_TENSORFLOW["no_inputs"] = (
    None,
    np.array([1.308330e-01, 1.056049e-17, -1.806317e00, -1.182600e-16]),
)


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

def gradient_test_setup():
    


@pytest.mark.parametrize("nshots", [None, 12000000])
@pytest.mark.parametrize("backend", EXECUTION_BACKENDS)
@pytest.mark.parametrize("wrt_inputs", [True, False])
@pytest.mark.parametrize("diff_rule", DIFF_RULES)
@pytest.mark.parametrize("equivariant", [True, False])
def test_expval_custom_grad(
    frontend,
    backend,
    nshots,
    wrt_inputs,
    diff_rule,
    equivariant,
):
    """
    Compute test gradient of < 0 | model^dag observable model | 0 > w.r.t model's
    parameters. In this test the system size is fixed to two qubits and all the
    parameters/data values are fixed.
    """

    if diff_rule is not None and diff_rule.__name__ == "Jax" and nshots is not None:
        pytest.skip("Jax differentiation does not work with shots.")

    seed = 42
    set_seed(frontend, seed)
    backend.set_seed(seed)

    x = construct_x(frontend, wrt_inputs=wrt_inputs)

    nqubits = 2
    nlayers = 1

    obs = hamiltonians.Z(nqubits=nqubits, backend=backend)

    encoding_layer = PhaseEncoding(nqubits=nqubits)
    training_layer = hardware_efficient(nqubits=nqubits, nlayers=nlayers, seed=seed)

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
        training_layer,
    ]
    if equivariant:
        circuit_structure += [
            equivariant_circuit,
        ]

    decoding_layer = Expectation(
        nqubits=nqubits,
        backend=backend,
        observable=obs,
        nshots=nshots,
    )

    nparams = len(training_layer.get_parameters())
    initial_params = np.linspace(0.0, 2 * np.pi, nparams)
    training_layer.set_parameters(
        backend.cast(initial_params, dtype=backend.np.float64)
    )

    q_model = frontend.QuantumModel(
        circuit_structure=circuit_structure,
        decoding=decoding_layer,
        differentiation=diff_rule,
    )

    grad_string = ""
    if equivariant:
        grad_string += "equivariant_"
    if wrt_inputs:
        grad_string += "wrt_inputs"
    else:
        grad_string += "no_inputs"
    print(grad_string)
    TARGET_GRAD = (
        TARGET_GRAD_TORCH
        if frontend.__name__ == "qiboml.interfaces.pytorch"
        else TARGET_GRAD_TENSORFLOW
    )
    input_target, params_target = TARGET_GRAD[grad_string]
    grad_wrt_input, grad_wrt_params = compute_gradient(
        frontend, q_model, x, wrt_inputs=wrt_inputs
    )
    grad_wrt_input = np.array(grad_wrt_input)
    grad_wrt_params = np.array(grad_wrt_params)
    print(grad_wrt_input)
    print(grad_wrt_params)
    tol = 1e-3 if nshots is None else 1e-1
    if input_target is None:
        assert grad_wrt_input.item() is None
    else:
        backend.assert_allclose(grad_wrt_input, input_target, atol=tol)
    backend.assert_allclose(grad_wrt_params, params_target, atol=tol)


TARGET_GRAD_REUPLOADING = {
    "wrt_inputs": (
        np.array([1.668244, -0.350845]),
        np.array(
            [
                0.553438,
                0.701541,
                -0.897603,
                0.369002,
                0.280684,
                0.303648,
                0.72218,
                -0.120478,
                0.679402,
                0.017968,
                2.286109,
            ]
        ),
    ),
    "no_inputs": (
        None,
        np.array(
            [
                0.182884,
                0.226752,
                -0.637248,
                -0.270067,
                0.315737,
                0.653579,
                0.899997,
                -0.235639,
                0.514433,
                -0.488684,
                1.068054,
            ]
        ),
    ),
}


@pytest.mark.parametrize("nshots", [None, 12000000])
#@pytest.mark.parametrize("backend", EXECUTION_BACKENDS)
#@pytest.mark.parametrize("diff_rule", DIFF_RULES)
@pytest.mark.parametrize("wrt_inputs", [True, False])
def test_expval_custom_grad_reuploading(
    #frontend,
    #backend,
    nshots,
    #diff_rule,
    wrt_inputs,
):
    """
    Compute test gradient of < 0 | model^dag observable model | 0 > w.r.t model's
    parameters. In this test the system size is fixed to two qubits and all the
    parameters/data values are fixed.
    """
    diff_rule = None
    from qiboml.backends import PyTorchBackend, TensorflowBackend
    backend = PyTorchBackend()
    import qiboml.interfaces.pytorch as frontend
    #import qiboml.interfaces.tensorflow as frontend
    
    if diff_rule is not None and diff_rule.__name__ == "Jax" and nshots is not None:
        pytest.skip("Jax differentiation does not work with shots.")
    
    set_seed(frontend, 42)
    backend.set_seed(42)

    x = construct_x(frontend, wrt_inputs=wrt_inputs)

    nqubits = 2

    obs = hamiltonians.Z(nqubits=nqubits, backend=backend)

    encoding_layer = PhaseEncoding(nqubits=nqubits)
    training_layer_1 = hardware_efficient(nqubits=nqubits, nlayers=1)
    training_layer_2 = training_layer_1.copy(deep=True)
    nparams = len(training_layer_1.get_parameters())
    training_layer_1.set_parameters(
        backend.cast(np.linspace(0.0, 2 * np.pi, nparams), dtype=backend.np.float64)
    )
    training_layer_1.set_parameters(
        backend.cast(np.linspace(0.0, np.pi, nparams), dtype=backend.np.float64)
    )

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
    ]
    circuit_structure += [
        equivariant_circuit,
    ]

    decoding_layer = Expectation(
        nqubits=nqubits,
        backend=backend,
        observable=obs,
        nshots=nshots,
    )

    q_model = frontend.QuantumModel(
        circuit_structure=circuit_structure,
        decoding=decoding_layer,
        differentiation=diff_rule,
    )

    TARGET_GRAD = (
        TARGET_GRAD_REUPLOADING["wrt_inputs"]
        if wrt_inputs
        else TARGET_GRAD_REUPLOADING["no_inputs"]
    )
    input_target, params_target = TARGET_GRAD
    grad_wrt_input, grad_wrt_params = compute_gradient(
        frontend, q_model, x, wrt_inputs=wrt_inputs
    )
    grad_wrt_input = np.array(grad_wrt_input)
    grad_wrt_params = np.array(grad_wrt_params)
    print(grad_wrt_input)
    print(grad_wrt_params)
    tol = 1e-3 if nshots is None else 1e-1
    if input_target is None:
        assert grad_wrt_input.item() is None
    else:
        backend.assert_allclose(grad_wrt_input, input_target, atol=tol)
    backend.assert_allclose(grad_wrt_params, params_target, atol=tol)
