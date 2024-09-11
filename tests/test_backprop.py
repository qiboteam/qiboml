import pytest
import torch
from qibo import hamiltonians
from qibo.backends import NumpyBackend, PyTorchBackend
from qibo.symbols import Z

from qiboml import pytorch as pt
from qiboml.backends import JaxBackend
from qiboml.models import ansatze as ans
from qiboml.models import encoding_decoding as ed


@pytest.mark.parametrize("backend", [JaxBackend(), PyTorchBackend()])
@pytest.mark.parametrize("differentiation", ["Jax", "PSR"])
def test_backpropagation(backend, differentiation):
    nqubits = 5
    dim = 4
    training_layer = ans.ReuploadingLayer(nqubits, backend=backend)
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
            training_layer,
            decoding_layer,
        ],
        differentiation=differentiation,
    )
    encoding = torch.nn.Linear(1, nqubits)
    model = torch.nn.Sequential(encoding, q_model)
    # try to fit a parabola
    x = torch.randn(10, 1)
    y = x**2

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
    for input, target in zip(x, y):
        optimizer.zero_grad()
        output = model(input)
        loss = (target - output) ** 2
        params_bkp = torch.cat(tuple(p.ravel() for p in model.parameters()))
        print(f"> loss: {loss}")
        loss.backward()
        optimizer.step()
        print(
            f"> Parameters delta: {torch.cat(tuple(p.ravel() for p in model.parameters())) - params_bkp}"
        )

    assert False
