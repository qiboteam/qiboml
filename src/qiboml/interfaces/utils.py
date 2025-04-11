from typing import Dict, List, Optional, Union

from qibo import Circuit

from qiboml import ndarray
from qiboml.models.decoding import QuantumDecoding
from qiboml.models.encoding import QuantumEncoding
from qiboml.operations.differentiation import PSR


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

    circuit = Circuit(circuit_structure[0].nqubits)
    for circ in circuit_structure:
        if isinstance(circ, QuantumEncoding):
            circuit += circ(x)
        else:
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
        differentiation = PSR()
    else:
        diff = instructions[backend_string]
        differentiation = diff() if diff is not None else None

    return differentiation
