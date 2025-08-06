import random
from inspect import signature
from typing import Callable, Dict, List, Optional, Tuple, Union

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


def circuit_from_structure(
    circuit_structure, params: ndarray, engine, x: Optional[ndarray] = None, tracer=None
) -> Tuple[Circuit, ndarray, ndarray]:
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

    # the complete circuit
    circuit = None
    # the jacobian for each sub-circuit
    jacobians = []
    jacobians_wrt_inputs = []
    input_to_gate_map = {}
    param_to_gate_map = {}
    dtype = params.dtype

    index = 0
    for circ in circuit_structure:
        if isinstance(circ, QuantumEncoding):
            if tracer is not None:
                jacobian, input_map, circ = tracer(circ, x)
                # update the input_map to the index of the global circuit
                input_map = {
                    inp: tuple(i + index for i in indices)
                    for inp, indices in input_map.items()
                }
                # update the global map
                for inp, indices in input_map.items():
                    if inp in input_to_gate_map:
                        input_to_gate_map[inp] += indices
                    else:
                        input_to_gate_map[inp] = indices
            else:
                jacobian, _, circ = None, None, circ(x)
            jacobians_wrt_inputs.append(jacobian)
        else:
            if isinstance(circ, Circuit):
                nparams = len(circ.get_parameters())
                circ.set_parameters(params[index : index + nparams])
                jacobian = engine.eye(nparams, dtype=dtype)
            elif isinstance(circ, Callable):
                param_dict = signature(circ).parameters
                nparams = len(param_dict)
                if tracer is not None:
                    jacobian, par_map, circ = tracer(
                        circ, params[index : index + nparams]
                    )
                else:
                    jacobian, par_map, circ = (
                        None,
                        None,
                        circ(*params[index : index + nparams]),
                    )
            index += nparams
            jacobians.append(jacobian)
        if circuit is None:
            circuit = circ
        else:
            circuit += circ

    # pad the jacobians to the total dimension (total_number_of_gates, total_number_of_parameters)
    if tracer is not None:
        total_dim = tuple(sum(np.array(j.shape) for j in jacobians))
        # build the global jacobian
        J = engine.zeros(total_dim, dtype=dtype)
        position = np.array([0, 0])
        # insert each sub-jacobian in the total one
        for j in jacobians:
            shape = np.array(j.shape)
            interval = tuple(zip(position, shape + position))
            J[interval[0][0] : interval[0][1], interval[1][0] : interval[1][1]] = j
            position += shape
    else:
        J = None
    return circuit, engine.vstack(jacobians_wrt_inputs), J, input_to_gate_map


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
        # differentiation = diff if diff is not None else None

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
