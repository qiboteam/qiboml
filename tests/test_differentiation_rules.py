import numpy as np
import pytest
from qibo import hamiltonians
from qibo.backends import NumpyBackend
from qibojit.backends import NumbaBackend

from qiboml.backends import PyTorchBackend
from qiboml.models.ansatze import HardwareEfficient
from qiboml.models.decoding import Expectation
from qiboml.models.encoding import PhaseEncoding
from qiboml.operations.differentiation import PSR, Jax

# TODO: use the classical conftest mechanism or customize mechanism for this test
EXECUTION_BACKENDS = [NumbaBackend(), NumpyBackend(), PyTorchBackend()]
DIFF_RULES = [Jax, PSR]

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
@pytest.mark.parametrize("diff_rule", DIFF_RULES)
def test_expval_custom_grad(frontend, backend, nshots, wrt_inputs, diff_rule):
    """
    Compute test gradient of < 0 | model^dag observable model | 0 > w.r.t model's
    parameters. In this test the system size is fixed to two qubits and all the
    parameters/data values are fixed.
    """

    if diff_rule.__name__ == "Jax" and nshots is not None:
        pytest.skip("Jax differentiation does not work with shots.")

    set_seed(frontend, 42)
    backend.set_seed(42)

    x = construct_x(frontend, with_factor=wrt_inputs)

    nqubits = 2

    obs = hamiltonians.Z(nqubits=nqubits, backend=backend)

    encoding_layer = PhaseEncoding(nqubits=nqubits)
    training_layer = HardwareEfficient(nqubits=nqubits, nlayers=1)

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
        circuit_structure=[encoding_layer, training_layer],
        decoding=decoding_layer,
        differentiation=diff_rule(),
    )

    target_grad = TARGET_GRAD["wrt_inputs"] if wrt_inputs else TARGET_GRAD["no_inputs"]
    grad = compute_gradient(frontend, q_model, x)
    tol = 1e-6 if nshots is None else 1e-1
    backend.assert_allclose(grad, target_grad, atol=tol)
