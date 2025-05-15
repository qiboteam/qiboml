from cuquantum import contract, contract_path

# Reference: https://github.com/NVIDIA/cuQuantum/blob/main/python/samples/cutensornet/tn_algorithms/mps_algorithms.ipynb


class MPSContractionHelper:
    """A helper class to compute various quantities for a given MPS.

    Interleaved format is used to construct the input args for `cuquantum.contract`.

    Reference: https://github.com/NVIDIA/cuQuantum/blob/main/python/samples/cutensornet/tn_algorithms/mps_algorithms.ipynb

    The following compute quantities are supported:

        - the norm of the MPS.
        - the equivalent state vector from the MPS.
        - the expectation value for a given operator.
        - the equivalent state vector after multiplying an MPO to an MPS.

    Parameters:
        num_qubits: The number of qubits for the MPS.
    """

    def __init__(self, num_qubits):
        self.num_qubits = num_qubits
        self.bra_modes = [(2 * i, 2 * i + 1, 2 * i + 2) for i in range(num_qubits)]
        offset = 2 * num_qubits + 1
        self.ket_modes = [
            (i + offset, 2 * i + 1, i + 1 + offset) for i in range(num_qubits)
        ]

    def contract_norm(self, mps_tensors, options=None):
        """Contract the corresponding tensor network to form the norm of the
        MPS.

        Parameters:
            mps_tensors: A list of rank-3 ndarray-like tensor objects.
                The indices of the ith tensor are expected to be bonding index to the i-1 tensor,
                the physical mode, and then the bonding index to the i+1th tensor.
            options: Specify the contract and decompose options.

        Returns:
            The norm of the MPS.
        """
        interleaved_inputs = []
        for i, o in enumerate(mps_tensors):
            interleaved_inputs.extend(
                [o, self.bra_modes[i], o.conj(), self.ket_modes[i]]
            )
        interleaved_inputs.append([])  # output
        return self._contract(interleaved_inputs, options=options).real

    def contract_state_vector(self, mps_tensors, options=None):
        """Contract the corresponding tensor network to form the state vector
        representation of the MPS.

        Parameters:
            mps_tensors: A list of rank-3 ndarray-like tensor objects.
                The indices of the ith tensor are expected to be bonding index to the i-1 tensor,
                the physical mode, and then the bonding index to the i+1th tensor.
            options: Specify the contract and decompose options.

        Returns:
            An ndarray-like object as the state vector.
        """
        interleaved_inputs = []
        for i, o in enumerate(mps_tensors):
            interleaved_inputs.extend([o, self.bra_modes[i]])
        output_modes = tuple([bra_modes[1] for bra_modes in self.bra_modes])
        interleaved_inputs.append(output_modes)  # output
        return self._contract(interleaved_inputs, options=options)

    def contract_expectation(
        self, mps_tensors, operator, qubits, options=None, normalize=False
    ):
        """Contract the corresponding tensor network to form the expectation of
        the MPS.

        Parameters:
            mps_tensors: A list of rank-3 ndarray-like tensor objects.
                The indices of the ith tensor are expected to be bonding index to the i-1 tensor,
                the physical mode, and then the bonding index to the i+1th tensor.
            operator: A ndarray-like tensor object.
                The modes of the operator are expected to be output qubits followed by input qubits, e.g,
                ``A, B, a, b`` where `a, b` denotes the inputs and `A, B'` denotes the outputs.
            qubits: A sequence of integers specifying the qubits that the operator is acting on.
            options: Specify the contract and decompose options.
            normalize: Whether to scale the expectation value by the normalization factor.

        Returns:
            An ndarray-like object as the state vector.
        """

        interleaved_inputs = []
        extra_mode = 3 * self.num_qubits + 2
        operator_modes = [None] * len(qubits) + [self.bra_modes[q][1] for q in qubits]
        qubits = list(qubits)
        for i, o in enumerate(mps_tensors):
            interleaved_inputs.extend([o, self.bra_modes[i]])
            k_modes = self.ket_modes[i]
            if i in qubits:
                k_modes = (k_modes[0], extra_mode, k_modes[2])
                q = qubits.index(i)
                operator_modes[q] = extra_mode  # output modes
                extra_mode += 1
            interleaved_inputs.extend([o.conj(), k_modes])
        interleaved_inputs.extend([operator, tuple(operator_modes)])
        interleaved_inputs.append([])  # output
        if normalize:
            norm = self.contract_norm(mps_tensors, options=options)
        else:
            norm = 1
        return self._contract(interleaved_inputs, options=options) / norm

    def _contract(self, interleaved_inputs, options=None):
        path = contract_path(*interleaved_inputs, options=options)[0]

        return contract(*interleaved_inputs, options=options, optimize={"path": path})
