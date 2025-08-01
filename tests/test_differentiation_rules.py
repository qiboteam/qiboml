import random

import numpy as np
import pytest
from qibo import Circuit, gates, hamiltonians
from qibo.backends import NumpyBackend
from qibojit.backends import NumbaBackend

from qiboml.backends import PyTorchBackend
from qiboml.models.ansatze import HardwareEfficient
from qiboml.models.decoding import Expectation
from qiboml.models.encoding import PhaseEncoding, QuantumEncoding
from qiboml.operations.differentiation import PSR, Jax

# TODO: use the classical conftest mechanism or customize mechanism for this test
EXECUTION_BACKENDS = [NumbaBackend(), NumpyBackend(), PyTorchBackend()]
DIFF_RULES = [Jax, PSR]

TARGET_GRAD = np.array([0.130832955241203, 0.0, -1.806316614151001, 0.0])
TARGET_GRAD = {
    "no_inputs": np.array([0.130832955241203, 0.0, -1.806316614151001, 0.0]),
    "wrt_inputs": np.array([0.740709794, 0.0, -0.730872325, 0.0]),
    "equivariant_no_inputs": np.array(
        [
            -0.30554144,
            -0.12198332,
            -0.1466465,
            -0.15876406,
            0.13876658,
            -0.05346034,
            0.70203977,
        ]
    ),
    "equivariant_wrt_inputs": np.array(
        [
            -0.06106278,
            0.14663098,
            0.00817663,
            -0.13743929,
            0.56005389,
            -0.15179269,
            1.77934261,
        ]
    ),
}


def set_seed(frontend, seed):
    random.seed(42)
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
@pytest.mark.parametrize("equivariant", [True, False])
def test_expval_custom_grad(
    frontend, backend, nshots, wrt_inputs, diff_rule, equivariant
):
    """
    Compute test gradient of < 0 | model^dag observable model | 0 > w.r.t model's
    parameters. In this test the system size is fixed to two qubits and all the
    parameters/data values are fixed.
    """

    if diff_rule.__name__ == "Jax" and nshots is not None:
        pytest.skip("Jax differentiation does not work with shots.")

    if equivariant and diff_rule.__name__ == "PSR":
        pytest.skip("PSR not supported with equivariant models yet.")

    set_seed(frontend, 42)
    backend.set_seed(42)

    x = construct_x(frontend, with_factor=wrt_inputs)

    nqubits = 2

    obs = hamiltonians.Z(nqubits=nqubits, backend=backend)

    encoding_layer = PhaseEncoding(nqubits=nqubits)
    training_layer = HardwareEfficient(nqubits=nqubits, nlayers=1)

    def equivariant_circuit(engine, th, phi, lam):
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
        differentiation=diff_rule(),
    )

    grad_string = ""
    if equivariant:
        grad_string += "equivariant_"
    if wrt_inputs:
        grad_string += "wrt_inputs"
    else:
        grad_string += "no_inputs"
    target_grad = TARGET_GRAD[grad_string]
    grad = compute_gradient(frontend, q_model, x)
    tol = 1e-6 if nshots is None else 1e-1
    backend.assert_allclose(grad, target_grad, atol=tol)
