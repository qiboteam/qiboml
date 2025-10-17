import numpy as np
import pytest
from qibo import gates

import qiboml.models.encoding as ed


def test_binary_encoding_layer(backend):
    nqubits = 10
    qubits = np.random.choice(range(nqubits), size=(6,), replace=False)
    layer = ed.BinaryEncoding(nqubits, qubits=qubits)
    data = backend.cast(np.random.choice([0, 1], size=(len(qubits),)))
    c = layer(data)
    for bit, gate in zip(data, c.queue):
        assert bit == gate.init_kwargs["theta"] / np.pi
    # test shape error
    with pytest.raises(RuntimeError):
        layer(backend.cast(np.random.choice([0, 1], size=(len(qubits) - 1,))))


def test_phase_encoding_layer(backend):
    nqubits = 10
    qubits = np.random.choice(range(nqubits), size=(6,), replace=False)

    # Testing error when encoding gate is affected by more than one parameter
    with pytest.raises(NotImplementedError):
        layer = ed.PhaseEncoding(nqubits=nqubits, qubits=qubits, encoding_gate=gates.U3)

    layer = ed.PhaseEncoding(nqubits, qubits=qubits)
    data = backend.cast(np.random.randn(1, len(qubits)))
    c = layer(data)
    angles = [gate.init_kwargs["theta"] for gate in c.queue if gate.name == "ry"]
    backend.assert_allclose(data.ravel(), angles)
