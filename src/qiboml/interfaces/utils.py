from typing import Dict, List, Optional, Union

import numpy as np
from qibo import Circuit
from qibo.ui.mpldrawer import plot_circuit

from qiboml import ndarray
from qiboml.models.decoding import QuantumDecoding
from qiboml.models.encoding import QuantumEncoding


def get_params_from_circuit_structure(
    circuit_structure: Union[Circuit, List[Union[Circuit, QuantumEncoding]]]
):
    """
    Helper function to retrieve the list of trainable parameters of a circuit
    given the circuit structure.
    """
    params = []
    for circ in circuit_structure:
        if not isinstance(circ, QuantumEncoding):
            params.extend([p for param in circ.get_parameters() for p in param])
    return params


def circuit_from_structure(
    circuit_structure,
    x: Optional[ndarray],
):
    """
    Helper function to reconstruct the whole circuit from a circuit structure.
    In the case the circuit structure involves encodings, the encoding data has
    to be provided as well.
    """

    if (
        any(isinstance(circ, QuantumEncoding) for circ in circuit_structure)
        and x is None
    ):
        raise ValueError(
            "x cannot be None when encoding layers are present in the circuit structure."
        )

    circuit = Circuit(
        circuit_structure[0].nqubits, density_matrix=circuit_structure[0].density_matrix
    )
    for circ in circuit_structure:
        if isinstance(circ, QuantumEncoding):
            circ = circ(x)
        circuit += circ
    return circuit


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

        differentiation = PSR()
    else:
        diff = instructions[backend_string]
        differentiation = diff() if diff is not None else None

    return differentiation


def draw_circuit(circuit_structure, backend, plt_drawing=True, **plt_kwargs):
    """
    Draw the full circuit structure.

    Args:
        plt_drawing (bool): if True, the `qibo.ui.plot_circuit` function is used.
            If False, the default `circuit.draw` method is used.
        plt_kwargs (dict): extra arguments which can be set to customize the
            `qibo.ui.plot_circuit` function.
    """

    encoding_layer = next(
        (circ for circ in circuit_structure if isinstance(circ, QuantumEncoding)),
        None,
    )
    dummy_data = (
        backend.cast(np.zeros(len(encoding_layer.qubits)))
        if encoding_layer is not None
        else None
    )
    circuit = circuit_from_structure(circuit_structure, dummy_data)
    if plt_drawing:
        _, fig = plot_circuit(circuit, **plt_kwargs)
        return fig
    else:
        circuit.draw()
        return str(circuit)


def _uniform_circuit_structure_density_matrix(circuit_structure):
    """
    Align the ``density_matrix`` attribute of all circuits composing the circuit structure.
    Namely, setting them to ``True`` if at least one component of the circuit has ``density_matrix==True``.
    """
    density_matrix = any(circ.density_matrix for circ in circuit_structure)
    for circ in circuit_structure:
        circ.density_matrix = density_matrix
