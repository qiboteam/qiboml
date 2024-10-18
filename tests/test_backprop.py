import matplotlib.pyplot as plt
import pytest
import torch
from qibo import hamiltonians
from qibo.backends import NumpyBackend, PyTorchBackend
from qibo.symbols import Z

from qiboml import pytorch as pt
from qiboml.backends import JaxBackend
from qiboml.models import ansatze as ans
from qiboml.models import encoding_decoding as ed


def eval_model(model, data, target, loss_f):
    with torch.no_grad():
        loss = 0.0
        for x, y in zip(data, target):
            out = model(x)
            loss += loss_f(out.ravel(), y.ravel())
        avg_loss = loss / data.shape[0]
        print(f"Avg loss: {avg_loss}")
        plt.scatter(data[:, 0], target, alpha=0.5)
        plt.scatter(data[:, 0], [model(x) for x in data], alpha=0.5)
        # plt.show()
    return avg_loss


@pytest.mark.parametrize(
    "backend",
    [
        PyTorchBackend(),
    ],
)
@pytest.mark.parametrize("differentiation", ["Jax", "PSR"])
def test_backpropagation(backend, differentiation):
    nqubits = 3
    training_layers = [ans.ReuploadingLayer(nqubits, backend=backend) for _ in range(3)]
    encoding_layer = ed.PhaseEncodingLayer(nqubits, backend=backend)
    kwargs = {"backend": backend}
    decoding_qubits = range(nqubits)
    observable = hamiltonians.SymbolicHamiltonian(
        sum([Z(int(i)) for i in decoding_qubits]),
        nqubits=nqubits,
        backend=backend,
    )
    kwargs["observable"] = observable
    kwargs["analytic"] = True
    decoding_layer = ed.ExpectationLayer(nqubits, decoding_qubits, **kwargs)
    q_model = pt.QuantumModel(
        layers=[
            encoding_layer,
            *training_layers,
            decoding_layer,
        ],
        differentiation=differentiation,
    )
    classical_encoding = torch.nn.Linear(1, nqubits).double()
    classical_decoding = torch.nn.Linear(1, 1).double()
    """
    model = torch.nn.Sequential(
        classical_encoding,
        torch.nn.ReLU(),
        q_model,
        torch.nn.ReLU(),
        classical_decoding
    )"""
    model = q_model
    # try to fit a parabola
    x = torch.randn(100, 3).double()
    y = torch.sin(x.sum(-1))
    loss_f = torch.nn.MSELoss()

    initial_loss = eval_model(model, x, y, loss_f)

    optimizer = torch.optim.Adam(model.parameters())
    cum_loss = 0.0

    for i, (input, target) in enumerate(zip(x, y)):
        optimizer.zero_grad()
        output = model(input)
        loss = loss_f(target, output)
        loss.backward()
        optimizer.step()

        cum_loss += loss
        if i % 50 == 0 and i != 0:
            print(f"Loss: {cum_loss / 50}")
            cum_loss = 0.0

    final_loss = eval_model(model, x, y, loss_f)

    assert initial_loss > final_loss
    assert False
