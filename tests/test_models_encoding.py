import numpy as np

import qiboml.models.encoding_decoding as ed


def test_binary_encoding_layer(backend):
    nqubits = 10
    qubits = np.random.choice(range(nqubits), size=(6,), replace=False)
    layer = ed.BinaryEncodingLayer(nqubits, qubits=qubits, backend=backend)
    data = backend.cast(np.random.choice([0, 1], size=(len(qubits),)))
    c = layer(data)
    indices = [gate.qubits[0] for gate in c.queue if gate.name == "x"]
    assert [qubits[i] for i in np.flatnonzero(data == 1)] == indices


def test_phase_encoding_layer(backend):
    nqubits = 10
    qubits = np.random.choice(range(nqubits), size=(6,), replace=False)
    layer = ed.PhaseEncodingLayer(nqubits, qubits=qubits, backend=backend)
    data = backend.cast(np.random.randn(len(qubits)))
    c = layer(data)
    angles = [gate.init_kwargs["theta"] for gate in c.queue if gate.name == "rz"]
    backend.assert_allclose(data, angles)
