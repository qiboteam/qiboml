import numpy as np
from qibo.backends import NumpyBackend
from qibo.config import raise_error
from qibo.result import QuantumState

from qibotn.backends.abstract import QibotnBackend

CUDA_TYPES = {}


class CuTensorNet(QibotnBackend, NumpyBackend):  # pragma: no cover
    # CI does not test for GPU
    """Creates CuQuantum backend for QiboTN."""

    def __init__(self, runcard):
        super().__init__()
        from cuquantum import cutensornet as cutn  # pylint: disable=import-error

        if runcard is not None:
            self.MPI_enabled = runcard.get("MPI_enabled", False)
            self.NCCL_enabled = runcard.get("NCCL_enabled", False)

            expectation_enabled_value = runcard.get("expectation_enabled")
            if expectation_enabled_value is True:
                self.expectation_enabled = True
                self.pauli_string_pattern = "XXXZ"
            elif expectation_enabled_value is False:
                self.expectation_enabled = False
            elif isinstance(expectation_enabled_value, dict):
                self.expectation_enabled = True
                expectation_enabled_dict = runcard.get("expectation_enabled", {})
                self.pauli_string_pattern = expectation_enabled_dict.get(
                    "pauli_string_pattern", None
                )
            else:
                raise TypeError("expectation_enabled has an unexpected type")

            mps_enabled_value = runcard.get("MPS_enabled")
            if mps_enabled_value is True:
                self.MPS_enabled = True
                self.gate_algo = {
                    "qr_method": False,
                    "svd_method": {
                        "partition": "UV",
                        "abs_cutoff": 1e-12,
                    },
                }
            elif mps_enabled_value is False:
                self.MPS_enabled = False
            elif isinstance(mps_enabled_value, dict):
                self.MPS_enabled = True
                self.gate_algo = mps_enabled_value
            else:
                raise TypeError("MPS_enabled has an unexpected type")

        else:
            self.MPI_enabled = False
            self.MPS_enabled = False
            self.NCCL_enabled = False
            self.expectation_enabled = False

        self.name = "qibotn"
        self.cuquantum = cuquantum
        self.cutn = cutn
        self.platform = "cutensornet"
        self.versions["cuquantum"] = self.cuquantum.__version__
        self.supports_multigpu = True
        self.handle = self.cutn.create()

        global CUDA_TYPES
        CUDA_TYPES = {
            "complex64": (
                self.cuquantum.cudaDataType.CUDA_C_32F,
                self.cuquantum.ComputeType.COMPUTE_32F,
            ),
            "complex128": (
                self.cuquantum.cudaDataType.CUDA_C_64F,
                self.cuquantum.ComputeType.COMPUTE_64F,
            ),
        }

    def __del__(self):
        if hasattr(self, "cutn"):
            self.cutn.destroy(self.handle)

    def cuda_type(self, dtype="complex64"):
        """Get CUDA Type.

        Parameters:
            dtype (str, optional): Either single ("complex64") or double (complex128) precision. Defaults to "complex64".

        Returns:
            CUDA Type: tuple of cuquantum.cudaDataType and cuquantum.ComputeType
        """
        if dtype in CUDA_TYPES:
            return CUDA_TYPES[dtype]
        else:
            raise TypeError("Type can be either complex64 or complex128")

    def execute_circuit(
        self, circuit, initial_state=None, nshots=None, return_array=False
    ):  # pragma: no cover
        """Executes a quantum circuit using selected TN backend.

        Parameters:
            circuit (:class:`qibo.models.circuit.Circuit`): Circuit to execute.
            initial_state (:class:`qibo.models.circuit.Circuit`): Circuit to prepare the initial state.
                If ``None`` the default ``|00...0>`` state is used.

        Returns:
            QuantumState or numpy.ndarray: If `return_array` is False, returns a QuantumState object representing the quantum state. If `return_array` is True, returns a numpy array representing the quantum state.
        """

        import qibotn.eval as eval

        if initial_state is not None:
            raise_error(NotImplementedError, "QiboTN cannot support initial state.")

        if (
            self.MPI_enabled == False
            and self.MPS_enabled == False
            and self.NCCL_enabled == False
            and self.expectation_enabled == False
        ):
            state = eval.dense_vector_tn(circuit, self.dtype)
        elif (
            self.MPI_enabled == False
            and self.MPS_enabled == True
            and self.NCCL_enabled == False
            and self.expectation_enabled == False
        ):
            state = eval.dense_vector_mps(circuit, self.gate_algo, self.dtype)
        elif (
            self.MPI_enabled == True
            and self.MPS_enabled == False
            and self.NCCL_enabled == False
            and self.expectation_enabled == False
        ):
            state, rank = eval.dense_vector_tn_MPI(circuit, self.dtype, 32)
            if rank > 0:
                state = np.array(0)
        elif (
            self.MPI_enabled == False
            and self.MPS_enabled == False
            and self.NCCL_enabled == True
            and self.expectation_enabled == False
        ):
            state, rank = eval.dense_vector_tn_nccl(circuit, self.dtype, 32)
            if rank > 0:
                state = np.array(0)
        elif (
            self.MPI_enabled == False
            and self.MPS_enabled == False
            and self.NCCL_enabled == False
            and self.expectation_enabled == True
        ):
            state = eval.expectation_pauli_tn(
                circuit, self.dtype, self.pauli_string_pattern
            )
        elif (
            self.MPI_enabled == True
            and self.MPS_enabled == False
            and self.NCCL_enabled == False
            and self.expectation_enabled == True
        ):
            state, rank = eval.expectation_pauli_tn_MPI(
                circuit, self.dtype, self.pauli_string_pattern, 32
            )
            if rank > 0:
                state = np.array(0)
        elif (
            self.MPI_enabled == False
            and self.MPS_enabled == False
            and self.NCCL_enabled == True
            and self.expectation_enabled == True
        ):
            state, rank = eval.expectation_pauli_tn_nccl(
                circuit, self.dtype, self.pauli_string_pattern, 32
            )
            if rank > 0:
                state = np.array(0)
        else:
            raise_error(NotImplementedError, "Compute type not supported.")

        if return_array:
            return state.flatten()
        else:
            return QuantumState(state.flatten())
