import tempfile
from pathlib import Path

import networkx as nx
import numpy as np
import torch
from qibo import Circuit, gates, get_backend, get_transpiler, set_backend
from qibo.backends import _Global
from qibo.noise import NoiseModel, PauliError
from qibo.transpiler import NativeGates, Passes, Unroller
from qibocal.auto.runcard import Runcard
from qibocal.auto.task import Action
from qibocal.protocols.allxy import allxy
from qibolab._core.backends import QibolabBackend

from qiboml.interfaces.pytorch import QuantumModel
from qiboml.models.decoding import Expectation
from qiboml.models.utils import Calibrator
from qiboml.operations.differentiation import PSR


def _default_transpiler(backend):
    from qibo.transpiler.optimizer import Preprocessing
    from qibo.transpiler.pipeline import Passes
    from qibo.transpiler.router import Sabre
    from qibo.transpiler.unroller import NativeGates, Unroller

    qubits = [0, 1]
    natives = backend.natives
    connectivity_edges = backend.connectivity
    if qubits is not None and natives is not None:
        connectivity = (
            nx.Graph(connectivity_edges)
            if connectivity_edges is not None
            else nx.Graph()
        )
        connectivity.add_nodes_from(qubits)

        return Passes(
            connectivity=connectivity,
            passes=[
                Preprocessing(),
                Sabre(),
                Unroller(NativeGates[natives]),
            ],
        )
    return Passes(passes=[])


def _set_circuit():
    vqe_circ = Circuit(
        2,
    )
    vqe_circ.add(gates.RX(0, 3 * np.pi / 4, trainable=True))
    vqe_circ.add(gates.RX(1, np.pi / 4, trainable=True))
    vqe_circ.add(gates.CZ(0, 1))
    return vqe_circ


def test_calibrator():
    backend = QibolabBackend(platform="mock")
    transpiler = _default_transpiler(backend=backend)
    nqubits = 2
    epochs = 3
    vqe_circ = _set_circuit()

    wire_names = (i for i in range(nqubits))

    single_shot_action = Action(
        id="sgle_shot",
        operation="single_shot_classification",
        parameters={"nshots": 100},
    )
    allxy = Action(id="allxy", operation="allxy", parameters={"nshots": 100})
    runcard = Runcard(
        actions=[single_shot_action, allxy],
        targets=["0", "1"],
    )
    calibrator = Calibrator(
        runcard=runcard,
        backend=backend,
        path=Path(tempfile.gettempdir() + "/report_test"),
        trigger_shots=10,
    )
    dec = Expectation(
        nqubits=nqubits,
        nshots=1024,
        density_matrix=False,
        wire_names=wire_names,
        transpiler=transpiler,
        calibrator=calibrator,
    )
    model = QuantumModel(
        circuit_structure=vqe_circ, decoding=dec, differentiation=PSR()
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    for epoch in range(epochs):
        optimizer.zero_grad()
        cost = model()
        cost.backward()
        optimizer.step()

    assert dec.calibrator._history != None
