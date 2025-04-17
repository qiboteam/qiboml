import numpy as np
import pytest
from qibo import hamiltonians
from qibo.backends import NumpyBackend

from qiboml.backends import JaxBackend, PyTorchBackend, TensorflowBackend
from qiboml.models.ansatze import ReuploadingCircuit
from qiboml.models.decoding import Expectation
from qiboml.models.encoding import PhaseEncoding
from qiboml.operations.differentiation import PSR

# jax and tensorflow don't work with the PSR
# EXECUTION_BACKENDS = [JaxBackend(), PyTorchBackend(), TensorflowBackend()]
EXECUTION_BACKENDS = [NumpyBackend(), PyTorchBackend()]

TARGET_GRAD = np.array([0.130832955241203, 0.0, -1.806316614151001, 0.0])
TARGET_GRAD = {
    "no_inputs": np.array([0.130832955241203, 0.0, -1.806316614151001, 0.0]),
    "wrt_inputs": np.array([0.740709794, 0.0, -0.730872325, 0.0]),
}


def set_seed(frontend, seed):
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


def construct_x(frontend, with_factor=False):
    if frontend.__name__ == "qiboml.interfaces.pytorch":
        x = frontend.torch.tensor([[0.5, 0.8]])
        if with_factor:
            return frontend.torch.tensor(2.0, requires_grad=True) * x
        return x
    elif frontend.__name__ == "qiboml.interfaces.keras":
        if frontend.keras.backend.backend() == "tensorflow":
            x = frontend.tf.constant([[0.5, 0.8]])
            if with_factor:
                return frontend.tf.Variable(2.0) * x
            return x
        elif frontend.keras.backend.backend() == "pytorch":
            raise NotImplementedError
        else:
            raise NotImplementedError


def compute_gradient(frontend, model, x):
    if frontend.__name__ == "qiboml.interfaces.keras":
        if frontend.keras.backend.backend() == "tensorflow":
            with frontend.tf.GradientTape() as tape:
                model.circuit_parameters = frontend.tf.Variable(
                    model.circuit_parameters
                )
                tape.watch(x)
                expval = model(x)
            return tape.gradient(expval, model.circuit_parameters)
        elif frontend.keras.backend.backend() == "pytorch":
            raise NotImplementedError
        else:
            raise NotImplementedError

    elif frontend.__name__ == "qiboml.interfaces.pytorch":
        expval = model(x)
        expval.backward()
        grad = np.array(list(model.parameters())[-1].grad)
        return grad


@pytest.mark.parametrize("nshots", [None, 12000000])
@pytest.mark.parametrize("backend", EXECUTION_BACKENDS)
@pytest.mark.parametrize("wrt_inputs", [True, False])
def test_expval_grad_PSR(frontend, backend, nshots, wrt_inputs):
    """
    Compute test gradient of < 0 | model^dag observable model | 0 > w.r.t model's
    parameters. In this test the system size is fixed to two qubits and all the
    parameters/data values are fixed.
    """

    if frontend.__name__ == "qiboml.interfaces.keras":
        from qiboml.interfaces.keras import QuantumModel

    elif frontend.__name__ == "qiboml.interfaces.pytorch":
        from qiboml.interfaces.pytorch import QuantumModel

    set_seed(frontend, 42)
    backend.set_seed(42)

    x = construct_x(frontend, with_factor=wrt_inputs)

    nqubits = 2

    obs = hamiltonians.Z(nqubits=nqubits, backend=backend)

    encoding_layer = PhaseEncoding(nqubits=nqubits)
    training_layer = ReuploadingCircuit(nqubits=nqubits, nlayers=1)
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

    q_model = QuantumModel(
        encoding=encoding_layer,
        circuit=training_layer,
        decoding=decoding_layer,
        differentiation=PSR(),
    )

    target_grad = TARGET_GRAD["wrt_inputs"] if wrt_inputs else TARGET_GRAD["no_inputs"]
    grad = compute_gradient(frontend, q_model, x)
    tol = 1e-6 if nshots is None else 1e-1
    backend.assert_allclose(grad, target_grad, atol=tol)
