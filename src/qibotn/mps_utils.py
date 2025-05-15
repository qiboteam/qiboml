import cupy as cp
from cuquantum import contract
from cuquantum.cutensornet.experimental import contract_decompose


def initial(num_qubits, dtype):
    r"""Generate the MPS with an initial state of :math:`\ket{00...00}`

    Parameters:
        num_qubits: Number of qubits in the Quantum Circuit.
        dtype: Either single ("complex64") or double (complex128) precision.

    Returns:
        The initial MPS tensors.
    """
    state_tensor = cp.asarray([1, 0], dtype=dtype).reshape(1, 2, 1)
    mps_tensors = [state_tensor] * num_qubits
    return mps_tensors


def mps_site_right_swap(mps_tensors, i, **kwargs):
    """Perform the swap operation between the ith and i+1th MPS tensors.

    Parameters:
        mps_tensors: Tensors representing MPS
        i (int): index of the tensor to swap

    Returns:
        The updated MPS tensors.
    """
    # contraction followed by QR decomposition
    a, _, b = contract_decompose(
        "ipj,jqk->iqj,jpk",
        *mps_tensors[i : i + 2],
        algorithm=kwargs.get("algorithm", None),
        options=kwargs.get("options", None),
    )
    mps_tensors[i : i + 2] = (a, b)
    return mps_tensors


def apply_gate(mps_tensors, gate, qubits, **kwargs):
    """Apply the gate operand to the MPS tensors in-place.

    # Reference: https://github.com/NVIDIA/cuQuantum/blob/main/python/samples/cutensornet/tn_algorithms/mps_algorithms.ipynb

    Parameters:
        mps_tensors: A list of rank-3 ndarray-like tensor objects.
            The indices of the ith tensor are expected to be the bonding index to the i-1 tensor,
            the physical mode, and then the bonding index to the i+1th tensor.
        gate: A ndarray-like tensor object representing the gate operand.
            The modes of the gate is expected to be output qubits followed by input qubits, e.g,
            ``A, B, a, b`` where ``a, b`` denotes the inputs and ``A, B`` denotes the outputs.
        qubits: A sequence of integers denoting the qubits that the gate is applied onto.
        algorithm: The contract and decompose algorithm to use for gate application.
            Can be either a `dict` or a `ContractDecomposeAlgorithm`.
        options: Specify the contract and decompose options.

    Returns:
        The updated MPS tensors.
    """

    n_qubits = len(qubits)
    if n_qubits == 1:
        # single-qubit gate
        i = qubits[0]
        mps_tensors[i] = contract(
            "ipj,qp->iqj", mps_tensors[i], gate, options=kwargs.get("options", None)
        )  # in-place update
    elif n_qubits == 2:
        # two-qubit gate
        i, j = qubits
        if i > j:
            # swap qubits order
            return apply_gate(mps_tensors, gate.transpose(1, 0, 3, 2), (j, i), **kwargs)
        elif i + 1 == j:
            # two adjacent qubits
            a, _, b = contract_decompose(
                "ipj,jqk,rspq->irj,jsk",
                *mps_tensors[i : i + 2],
                gate,
                algorithm=kwargs.get("algorithm", None),
                options=kwargs.get("options", None),
            )
            mps_tensors[i : i + 2] = (a, b)  # in-place update
        else:
            # non-adjacent two-qubit gate
            # step 1: swap i with i+1
            mps_site_right_swap(mps_tensors, i, **kwargs)
            # step 2: apply gate to (i+1, j) pair. This amounts to a recursive swap until the two qubits are adjacent
            apply_gate(mps_tensors, gate, (i + 1, j), **kwargs)
            # step 3: swap back i and i+1
            mps_site_right_swap(mps_tensors, i, **kwargs)
    else:
        raise NotImplementedError("Only one- and two-qubit gates supported")
