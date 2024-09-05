import torch
from qibo import hamiltonians
from qibo.backends import NumpyBackend, PyTorchBackend
from qibo.symbols import Z

from qiboml import pytorch as pt
from qiboml.models import ansatze as ans
from qiboml.models import encoding_decoding as ed

# backend = PyTorchBackend()
backend = NumpyBackend()

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
    ]
)
print(list(q_model.parameters()))
data = torch.randn(1, 5)
data.requires_grad = True
out = q_model(data)
print(out.requires_grad)
loss = (out - 1.0) ** 2
print(loss.requires_grad)
loss.backward()
