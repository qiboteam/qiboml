import numpy as np
from qibo import gates, set_backend
from qibo.models import QFT
from qibo.quantum_info import random_clifford

import qiboml.models.encoding_decoding as ed

set_backend("numpy")


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


def test_probabilities_layer(backend):
    nqubits = 5
    qubits = np.random.choice(range(nqubits), size=(4,), replace=False)
    layer = ed.ProbabilitiesLayer(nqubits, qubits=qubits, backend=backend)
    c = random_clifford(nqubits, backend=backend)
    backend.assert_allclose(layer(c), c().probabilities(qubits))


def test_state_layer(backend):
    nqubits = 5
    layer = ed.StateLayer(nqubits, backend=backend)
    c = random_clifford(nqubits, backend=backend)
    real, im = layer(c)
    backend.assert_allclose(real + 1j * im, c().state())
