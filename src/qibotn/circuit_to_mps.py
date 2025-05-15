import cupy as cp
import numpy as np
from cuquantum import cutensornet as cutn

from qibotn.circuit_convertor import QiboCircuitToEinsum
from qibotn.mps_utils import apply_gate, initial


class QiboCircuitToMPS:
    """A helper class to convert Qibo circuit to MPS.

    Parameters:
        circ_qibo: The quantum circuit object.
        gate_algo(dict): Dictionary for SVD and QR settings.
        datatype (str): Either single ("complex64") or double (complex128) precision.
        rand_seed(int): Seed for random number generator.
    """

    def __init__(
        self,
        circ_qibo,
        gate_algo,
        dtype="complex128",
        rand_seed=0,
    ):
        np.random.seed(rand_seed)
        cp.random.seed(rand_seed)

        self.num_qubits = circ_qibo.nqubits
        self.handle = cutn.create()
        self.dtype = dtype
        self.mps_tensors = initial(self.num_qubits, dtype=dtype)
        circuitconvertor = QiboCircuitToEinsum(circ_qibo, dtype=dtype)

        for gate, qubits in circuitconvertor.gate_tensors:
            # mapping from qubits to qubit indices
            # apply the gate in-place
            apply_gate(
                self.mps_tensors,
                gate,
                qubits,
                algorithm=gate_algo,
                options={"handle": self.handle},
            )

    def __del__(self):
        cutn.destroy(self.handle)
