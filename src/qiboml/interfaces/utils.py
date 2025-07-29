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
    x: Optional[ndarray],
    params: Optional[ndarray],
    backend: Optional[Backend],
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
        # density_matrix=circuit_structure[0].density_matrix,
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
        elif not isinstance(circ, QuantumEncoding) and isinstance(circ, Callable):
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


def _uniform_circuit_structure(circuit_structure):
    """
    Align the ``density_matrix`` attribute of all circuits composing the circuit structure.
    Namely, setting their ``density_matrix=True`` if at least one component of the circuit has ``density_matrix==True``.
    """
    density_matrix = any(
        circ.density_matrix
        for circ in circuit_structure
        if not (isinstance(circ, Callable) and not isinstance(circ, QuantumEncoding))
    )
    for circ in circuit_structure:
        is_circuit = isinstance(circ, Circuit)
        is_encoding = isinstance(circ, QuantumEncoding)
        if is_circuit or is_encoding:
            circ.density_matrix = density_matrix
            if is_circuit:
                circ.init_kwargs["density_matrix"] = density_matrix


def _independent_params_map(params):
    """
    Extract the independent parameters among ``params`` by looking for all the
    elements of ``params`` that share the same memory (numpy.shares_memory).
    After that, a mapping is built that associates the index of an independent
    element with all the indices where it is repeated, for instance if ``params``
    is:

    array([0.1, 0.2, 0.1, 0.1, 0.2])

    the constructed map will be:

    {0: {0,2,3}, 1: {1,4}}

    (modulo the set ordering)
    """
    # the first element is surely independent, start from there
    imap = {0: {0}}
    # check all the other parameters
    for i, p1 in enumerate(params[1:], start=1):
        # check if any of the independent elements in imap share the
        # memory with the current element
        keys = [j for j in imap if np.shares_memory(p1, params[j])]
        # none found -> i (p1) is independent
        if len(keys) == 0:
            imap[i] = {i}
        # not independent, add i to the j (it should be only one) that
        # shares the memory with it
        else:
            for j in keys:
                imap[j].add(i)
    return imap


def set_parameters(circuit, params, imap):
    new_params = len(circuit.get_parameters()) * [
        None,
    ]
    for i in range(len(imap)):
        for j in imap[i]:
            new_params[j] = params[i]
    circuit.set_parameters(new_params)
