import random
from inspect import signature
from typing import Callable, Dict, List, Union

import numpy as np
from qibo import Circuit
from qibo.backends import _check_backend
from qibo.backends.abstract import Backend
from qibo.ui.mpldrawer import plot_circuit

from qiboml import ndarray
from qiboml.models.decoding import QuantumDecoding
from qiboml.models.encoding import QuantumEncoding


def get_params_from_circuit_structure(
    circuit_structure: List[Union[Circuit, QuantumEncoding, Callable]],
):
    """
    Helper function to retrieve the list of trainable parameters of a circuit
    given the circuit structure.
    """
    params = []
    for circ in circuit_structure:
        if isinstance(circ, Circuit):
            params.extend([p for param in circ.get_parameters() for p in param])
        elif not isinstance(circ, QuantumEncoding) and isinstance(circ, Callable):
            params.extend(
                random.random() for _ in range(len(signature(circ).parameters))
            )
    return params


def get_default_differentiation(decoding: QuantumDecoding, instructions: Dict):
    """
    Helper function to return default differentiation rule in case it is
    not set by the user.
    """
    backend_string = (
        f"{decoding.backend.name}-{decoding.backend.platform}"
        if decoding.backend.platform is not None
        else decoding.backend.name
    )

    if not decoding.analytic or backend_string not in instructions.keys():
        from qiboml.operations.differentiation import PSR

        differentiation = PSR
    else:
        differentiation = instructions[backend_string]

    return differentiation


def draw_circuit(model, plt_drawing=True, **plt_kwargs):
    """
    Draw the full circuit structure.

    Args:
        plt_drawing (bool): if True, the `qibo.ui.plot_circuit` function is used.
            If False, the default `circuit.draw` method is used.
        plt_kwargs (dict): extra arguments which can be set to customize the
            `qibo.ui.plot_circuit` function.
    """
    circuit_structure = model.circuit_structure
    backend = model.backend
    encoding_layer = next(
        (circ for circ in circuit_structure if isinstance(circ, QuantumEncoding)),
        None,
    )
    dummy_data = (
        backend.cast(np.zeros(len(encoding_layer.qubits)))
        if encoding_layer is not None
        else None
    )
    dummy_params = []
    for circ in circuit_structure:
        if isinstance(circ, Circuit):
            dummy_params.extend(len(circ.get_parameters()) * [0.0])
        elif not isinstance(circ, QuantumEncoding) and isinstance(circ, Callable):
            dummy_params.extend((len(signature(circ).parameters)) * [0.0])
    circuit = model.circuit_tracer.build_circuit(params=dummy_params, x=dummy_data)
    if plt_drawing:
        _, fig = plot_circuit(circuit, **plt_kwargs)
        return fig
    else:
        circuit.draw()
        return str(circuit)
