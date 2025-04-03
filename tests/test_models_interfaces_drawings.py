from pathlib import Path

import matplotlib
import matplotlib.pyplot
import numpy as np
import pytest
from qibo import Circuit, gates

from qiboml.models.decoding import Expectation
from qiboml.models.encoding import PhaseEncoding

matplotlib.use("agg")

BASEPATH = str(Path(__file__).parent / "plt_test_files")


def fig2array(fig):
    """Convert matplotlib image into numpy array."""
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    return data


def match_figure_image(fig, arr_path):
    """Check whether the two image arrays match."""
    return np.all(fig2array(fig) == np.load(arr_path))


def trainable_circuit(nqubits, entanglement=True):
    """Helper function to construct the trainable part of the model."""
    trainable_circ = Circuit(nqubits)
    for q in range(nqubits):
        trainable_circ.add(gates.RY(q=q, theta=0.0))
        trainable_circ.add(gates.RZ(q=q, theta=0.0))
    if nqubits > 1 and entanglement:
        [
            trainable_circ.add(
                gates.CNOT(q % nqubits, (q + 1) % nqubits) for q in range(nqubits)
            )
        ]
    return trainable_circ


@pytest.mark.parametrize("plt_drawing", [True, False])
def test_model_draw(frontend, plt_drawing):
    """Testing the ``model.draw`` feature."""

    target_drawing = "q0: ─RY─RY─RZ─o─X─\n" "q1: ─RY─RY─RZ─X─o─"

    nqubits = 2

    circuit_structure = [
        PhaseEncoding(nqubits=nqubits),
        trainable_circuit(nqubits=nqubits, entanglement=True),
    ]

    model = frontend.QuantumModel(
        circuit_structure=circuit_structure,
        decoding=Expectation(nqubits=nqubits),
    )

    fig = model.draw(plt_drawing=plt_drawing)

    if plt_drawing:
        match_figure_image(fig, BASEPATH + "/model_draw.npy")
    else:
        assert (fig, target_drawing)
