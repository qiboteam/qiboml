def _get_wire_names_and_qubits(nqubits, qubits):
    if qubits is not None:
        if isinstance(qubits[0], str):
            wire_names = qubits
            qubits = tuple(range(len(qubits)))
        else:
            qubits = tuple(qubits)
            wire_names = None
    else:
        qubits = tuple(range(nqubits))
        wire_names = None
    return qubits, wire_names
