import numpy as np
import pytest
import qibo
import torch
from qibo import hamiltonians
from qibo.backends import NumpyBackend, PyTorchBackend
from qibojit.backends import NumbaBackend

from qiboml.models.ansatze import ReuploadingCircuit
from qiboml.models.decoding import Expectation
from qiboml.models.encoding import PhaseEncoding
from qiboml.operations.differentiation import PSR

# TODO: use the classical conftest mechanism or customize mechanism for this test
EXECUTION_BACKENDS = [NumbaBackend(), NumpyBackend(), PyTorchBackend()]

TARGET_GRAD = np.array([0.130832955241203, 0.0, -1.806316614151001, 0.0])

torch.set_default_dtype(torch.float64)
torch.set_printoptions(precision=15, sci_mode=False)


def construct_x(frontend):
    if frontend.__name__ == "qiboml.interfaces.pytorch":
        return frontend.torch.tensor([0.5, 0.8])
    elif frontend.__name__ == "qiboml.interfaces.keras":
        return frontend.tf.Variable([0.5, 0.8])


def compute_gradient(frontend, model, x):
    if frontend.__name__ == "qiboml.interfaces.keras":
        # TODO: to check if this work once keras interface is introduced
        with frontend.tf.GradientTape() as tape:
            expval = model(x)
        return tape.gradient(expval, model.parameters)

    elif frontend.__name__ == "qiboml.interfaces.pytorch":
        expval = model(x)
        expval.backward()
        # TODO: standardize this output with keras' one and use less convolutions
        grad = np.array(list(model.parameters())[-1].grad)
        return grad


@pytest.mark.parametrize("nshots", [None, 1000000])
@pytest.mark.parametrize("backend", EXECUTION_BACKENDS)
def test_expval_grad_PSR(frontend, backend, nshots):
    """
    Compute test gradient of < 0 | model^dag observable model | 0 > w.r.t model's
    parameters. In this test the system size is fixed to two qubits and all the
    parameters/data values are fixed.
    """

    if frontend.__name__ == "qiboml.interfaces.keras":
        pytest.skip("keras interface not ready.")
    elif frontend.__name__ == "qiboml.interfaces.pytorch":
        # TODO: replace with qiboml, pytorch as soon as migration is complete
        # TODO: define a proper qiboml.set_interface() procedure for these situations
        qibo.set_backend("pytorch")

        from qiboml.interfaces.pytorch import QuantumModel

    decimals = 6 if nshots is None else 2

    x = construct_x(frontend)

    nqubits = 2

    obs = hamiltonians.Z(nqubits=nqubits)

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
    training_layer.set_parameters(initial_params)

    q_model = QuantumModel(
        encoding=encoding_layer,
        circuit=training_layer,
        decoding=decoding_layer,
        differentiation_rule=PSR(),
    )

    grad = compute_gradient(frontend, q_model, x)

    assert np.round(grad[0], decimals=decimals) == np.round(
        TARGET_GRAD[0], decimals=decimals
    )
    assert np.round(grad[2], decimals=decimals) == np.round(
        TARGET_GRAD[2], decimals=decimals
    )
