import random
from inspect import signature
from typing import Callable, Dict, List, Optional, Union

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
                random.random() for key in signature(circ).parameters if key != "engine"
            )
    return params


def circuit_from_structure(
    circuit_structure,
    x: Optional[ndarray] = None,
    params: Optional[ndarray] = None,
    backend: Optional[Backend] = None,
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

    backend = _check_backend(backend)

    circuit = Circuit(
        circuit_structure[0].nqubits,
    )
    index = 0
    for circ in circuit_structure:
        if isinstance(circ, QuantumEncoding):
            circ = circ(x)
        elif params is not None:
            if isinstance(circ, Circuit):
                nparams = len(circ.get_parameters())
                circ.set_parameters(params[index : index + nparams])
            elif isinstance(circ, Callable):
                param_dict = signature(circ).parameters
                nparams = len(param_dict)
                if "engine" in param_dict:
                    nparams -= 1
                circ = circ(backend.np, *params[index : index + nparams])
            else:
                raise RuntimeError
            index += nparams
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
    dummy_params = []
    for circ in circuit_structure:
        if isinstance(circ, Circuit):
            dummy_params.extend(len(circ.get_parameters()) * [0.0])
        elif not isinstance(circ, QuantumEncoding) and isinstance(
            circ, Callable
        ):  # pragma: no cover
            dummy_params.extend((len(signature(circ).parameters) - 1) * [0.0])
    circuit = circuit_from_structure(
        circuit_structure, x=dummy_data, params=dummy_params, backend=backend
    )
    if plt_drawing:
        _, fig = plot_circuit(circuit, **plt_kwargs)
        return fig
    else:
        circuit.draw()
        return str(circuit)
