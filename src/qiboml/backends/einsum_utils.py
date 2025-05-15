from functools import cache

from qibo.backends.einsum_utils import *


@cache
def prepare_strings(qubits, nqubits):
    if nqubits + len(qubits) + 1 > len(EINSUM_CHARS):  # pragma: no cover
        raise_error(NotImplementedError, "Not enough einsum characters.")

    inp = list(EINSUM_CHARS[: nqubits + 1])
    out = inp[:]
    trans = list(EINSUM_CHARS[nqubits + 1 : nqubits + len(qubits) + 1])
    for i, q in enumerate(qubits):
        q += 1
        trans.append(inp[q])
        out[q] = trans[i]

    inp = "".join(inp)
    out = "".join(out)
    trans = "".join(trans)
    rest = EINSUM_CHARS[nqubits + len(qubits) + 1 :]
    return inp, out, trans, rest


@cache
def control_order(control_qubits, target_qubits, nqubits):
    loop_start = 0
    order = list(control_qubits)
    targets = list(target_qubits)
    for control in control_qubits:
        for i in range(loop_start, control):
            order.append(i)
        loop_start = control + 1
        for i, t in enumerate(target_qubits):
            if t > control:
                targets[i] -= 1
    for i in range(loop_start, nqubits):
        order.append(i)
    for i in range(len(order)):
        order[i] += 1
        targets[i] += 1
    return order, targets
