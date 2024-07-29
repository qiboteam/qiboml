import numpy as np
from qibo import gates
from qibo.models import QFT
from qibo.quantum_info import random_clifford

import qiboml.models.encoding_decoding as ed


def test_probabilities_layer(backend):
    nqubits = 5
    qubits = np.random.choice(range(nqubits), size=(4,), replace=False)
    layer = ed.ProbabilitiesLayer(nqubits, qubits=qubits, backend=backend)
    c = random_clifford(nqubits, backend=backend)
    backend.assert_allclose(layer(c).ravel(), c().probabilities(qubits))


def test_state_layer(backend):
    nqubits = 5
    layer = ed.StateLayer(nqubits, backend=backend)
    c = random_clifford(nqubits, backend=backend)
    real, im = layer(c)
    backend.assert_allclose(real + 1j * im, c().state())
