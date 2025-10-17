"""Module defining the TensorFlow backend."""

import os
from collections import Counter
from typing import Optional, Union

import numpy as np
from qibo import __version__
from qibo.backends import Backend
from qibo.backends.npmatrices import NumpyMatrices
from qibo.config import TF_LOG_LEVEL, log, raise_error
from qibo.result import CircuitResult, QuantumState


class TensorflowMatrices(NumpyMatrices):
    # Redefine parametrized gate matrices for backpropagation to work

    def __init__(self, dtype):
        super().__init__(dtype)
        import tensorflow as tf  # pylint: disable=import-error,C0415

        self.engine = tf

    def _cast(self, x, dtype):
        return self.engine.cast(x, dtype=dtype)

    def Unitary(self, u):
        return self._cast(u, dtype=self.dtype)


class TensorflowBackend(Backend):
    def __init__(self):
        super().__init__()
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = str(TF_LOG_LEVEL)

        import tensorflow as tf  # pylint: disable=import-error,C0415
        from tensorflow.python.framework import (  # pylint: disable=E0611,C0415,import-error
            errors_impl,
        )

        self.engine = tf
        self.engine.experimental.numpy.experimental_enable_numpy_behavior()

        if TF_LOG_LEVEL >= 2:
            self.engine.get_logger().setLevel("ERROR")

        self.matrices = TensorflowMatrices(self.dtype)
        self.name = "qiboml"
        self.nthreads = 0
        self.oom_error = errors_impl.ResourceExhaustedError
        self.platform = "tensorflow"
        self.tensor_types = (np.ndarray, tf.Tensor, tf.Variable)
        self.versions = {
            "qibo": __version__,
            "numpy": np.__version__,
            "tensorflow": tf.__version__,
        }

        cpu_devices = self.engine.config.list_logical_devices("CPU")
        gpu_devices = self.engine.config.list_logical_devices("GPU")
        if gpu_devices:  # pragma: no cover
            # CI does not use GPUs
            self.device = gpu_devices[0].name
        elif cpu_devices:
            self.device = cpu_devices[0].name

    def cast(self, array, dtype=None, copy=False):
        if dtype is None:
            dtype = self.dtype

        array = self.engine.cast(array, dtype=dtype)

        if copy:
            return self.copy(array)

        return array

    def compile(self, func):
        return self.engine.function(func)

    def is_sparse(self, array):
        return isinstance(array, self.engine.sparse.SparseTensor)

    def set_device(self, device):  # pragma: no cover
        self.device = device

    def set_threads(self, nthreads):
        log.warning(
            "`set_threads` is not supported by the tensorflow "
            "backend. Please use tensorflow's thread setters: "
            "`tf.config.threading.set_inter_op_parallelism_threads` "
            "or `tf.config.threading.set_intra_op_parallelism_threads` "
            "to switch the number of threads."
        )

    def to_numpy(self, array):
        return np.array(array)

    ########################################################################################
    ######## Methods related to array manipulation                                  ########
    ########################################################################################

    def copy(self, array, **kwargs):
        return self.engine.identity(array, **kwargs)

    def default_rng(self, seed: Optional[int] = None):
        if seed is None:
            return self.engine.random.Generator.from_non_deterministic_state()

        return self.engine.random.Generator.from_seed(seed)

    def flatnonzero(self, array):
        return np.flatnonzero(array)

    def kron(self, array_1, array_2) -> "ndarray":
        return self.engine.experimental.numpy.kron(array_1, array_2)

    def matrix_norm(self, state, order: Union[int, float, str] = "nuc", **kwargs):
        state = self.cast(state, dtype=state.dtype)
        if order == "nuc":
            return self.trace(state)
        return self.engine.norm(state, ord=order, **kwargs)

    def real(self, array):
        return self.engine.math.real(array)

    def vector_norm(self, state, order: Union[int, float, str] = 2, dtype=None):
        if dtype is None:
            dtype = self.dtype

        state = self.cast(state, dtype=dtype)

        return self.engine.norm(state, ord=order)

    ########################################################################################
    ######## Methods related to linear algebra operations                           ########
    ########################################################################################

    def matrix_exp(
        self,
        matrix,
        phase: Union[float, int, complex] = 1,
        eigenvectors=None,
        eigenvalues=None,
    ):
        if eigenvectors is None or self.is_sparse(matrix):
            return self.expm(phase * matrix)
        return super().matrix_exp(matrix, phase, eigenvectors, eigenvalues)

    def matrix_power(
        self,
        matrix,
        power: Union[float, int],
        precision_singularity: float = 1e-14,
        dtype=None,
    ):
        if not isinstance(power, (float, int)):
            raise_error(
                TypeError,
                f"``power`` must be either float or int, but it is type {type(power)}.",
            )

        if dtype is None:
            dtype = self.dtype

        if power < 0.0:
            # negative powers of singular matrices via SVD
            determinant = self.det(matrix)
            if abs(determinant) < precision_singularity:
                return self._negative_power_singular_matrix(
                    matrix, power, precision_singularity, dtype
                )

        return super().matrix_power(matrix, power, precision_singularity)

    def singular_value_decomposition(self, array):
        # needed to unify order of return
        s_matrix, u_matrix, v_matrix = self.engine.linalg.svd(array)
        return u_matrix, s_matrix, self.conj(self.transpose(v_matrix))

    def jacobian(
        self, circuit, parameters=None, initial_state=None, return_complex: bool = True
    ):
        copied = circuit.copy(deep=True)

        # necessary for the tape to properly watch the variables
        parameters = self.engine.Variable(parameters)

        with self.engine.GradientTape(persistent=return_complex) as tape:
            copied.set_parameters(parameters)
            state = self.execute_circuit(copied, initial_state=initial_state).state()
            real = self.real(state)
            if return_complex:
                imag = self.imag(state)

        if return_complex:
            return tape.jacobian(real, parameters), tape.jacobian(imag, parameters)

        return tape.jacobian(real, parameters)

    def zero_state(self, nqubits, density_matrix: bool = False, dtype=None):
        if dtype is None:
            dtype = self.dtype

        shape = [[0, 0]] if density_matrix else [[0]]
        idx = self.engine.constant(shape, dtype="int32")

        shape = 2 * (2**nqubits,) if density_matrix else (2**nqubits,)
        state = self.zeros(shape, dtype=dtype)

        update = self.engine.constant([1], dtype=dtype)

        state = self.engine.tensor_scatter_nd_update(state, idx, update)

        return state

    ########################################################################################
    ######## Methods related to circuit execution                                   ########
    ########################################################################################

    def execute_circuit(self, circuit, initial_state=None, nshots=1000):
        with self.engine.device(self.device):
            return super().execute_circuit(circuit, initial_state, nshots)

    def execute_circuit_repeated(self, circuit, nshots, initial_state=None):
        with self.engine.device(self.device):
            return super().execute_circuit_repeated(circuit, nshots, initial_state)

    def matrix(self, gate):
        npmatrix = super().matrix(gate)
        # delete cached matrix if it's symbolic
        if self.engine.is_symbolic_tensor(npmatrix):
            delattr(self.matrices, gate.__class__.__name__)
        return npmatrix

    def matrix_fused(self, fgate):
        rank = len(fgate.target_qubits)
        # tf only supports coo sparse arrays
        # however they are probably not as efficient as csr ones
        # at this point it is maybe better to just use dense arrays
        # from the tests no major performance difference emerged
        # with dense tensors, thus keeping sparse representation for now
        matrix = self.identity(2**rank, dtype=self.dtype)

        for gate in fgate.gates:
            gmatrix = gate.matrix(self)
            # add controls if controls were instantiated using
            # the ``Gate.controlled_by`` method
            num_controls = len(gate.control_qubits)
            if num_controls > 0:
                gmatrix = self.engine.linalg.LinearOperatorBlockDiag(
                    self.identity(2 ** len(gate.qubits) - len(gmatrix)), gmatrix
                )
            # Kronecker product with identity is needed to make the
            # original matrix have shape (2**rank x 2**rank)
            eye = self.identity(2 ** (rank - len(gate.qubits)))
            gmatrix = self.kron(gmatrix, eye)
            # Transpose the new matrix indices so that it targets the
            # target qubits of the original gate
            original_shape = gmatrix.shape
            gmatrix = self.reshape(gmatrix, 2 * rank * (2,))
            qubits = list(gate.qubits)
            indices = qubits + [q for q in fgate.target_qubits if q not in qubits]
            indices = np.argsort(indices)
            transpose_indices = list(indices)
            transpose_indices.extend(indices + rank)
            gmatrix = self.transpose(gmatrix, transpose_indices)
            gmatrix = self.reshape(gmatrix, original_shape)
            # fuse the individual gate matrix to the total ``FusedGate`` matrix
            # we are using sparse matrices to improve perfomances
            matrix = self.engine.sparse.sparse_dense_matmul(
                self.engine.sparse.from_dense(gmatrix), matrix
            )

        return matrix

    def matrix_parametrized(self, gate):
        npmatrix = super().matrix_parametrized(gate)
        return self.cast(npmatrix, dtype=self.dtype)

    ########################################################################################
    ######## Methods related to the execution and post-processing of measurements   ########
    ########################################################################################

    def calculate_frequencies(self, samples):
        # redefining this because ``tnp.unique`` is not available
        res, _, counts = self.engine.unique_with_counts(samples, out_idx="int64")
        res, counts = np.array(res), np.array(counts)
        if res.dtype == "string":
            res = [r.numpy().decode("utf8") for r in res]
        else:
            res = [int(r) for r in res]
        return Counter({k: int(v) for k, v in zip(res, counts)})

    def sample_shots(self, probabilities, nshots):
        # redefining this because ``tnp.random.choice`` is not available
        logits = self.log(probabilities)[self.engine.newaxis]
        samples = self.engine.random.categorical(logits, nshots)[0]
        return samples

    def samples_to_binary(self, samples, nqubits):
        # redefining this because ``tnp.right_shift`` is not available
        qrange = self.engine.range(nqubits - 1, -1, -1, dtype=self.int32)
        samples = self.cast(samples, dtype=self.int32)
        samples = self.engine.bitwise.right_shift(samples[:, np.newaxis], qrange)
        return samples % 2

    def update_frequencies(self, frequencies, probabilities, nsamples):
        # redefining this because ``tnp.unique`` and tensor update is not available
        samples = self.sample_shots(probabilities, nsamples)
        res, _, counts = self.engine.unique_with_counts(samples, out_idx=self.int64)
        frequencies = self.engine.tensor_scatter_nd_add(
            frequencies, res[:, self.engine.newaxis], counts
        )
        return frequencies

    def assert_allclose(
        self, value, target, rtol: float = 1e-7, atol: float = 0.0
    ):  # pragma: no cover
        if isinstance(value, (CircuitResult, QuantumState)):
            value = value.state()
        if isinstance(target, (CircuitResult, QuantumState)):
            target = target.state()

        np.testing.assert_allclose(value, target, rtol=rtol, atol=atol)

    def _test_regressions(self, name):
        if name == "test_measurementresult_apply_bitflips":
            return [
                [4, 0, 0, 1, 0, 0, 1, 0, 0, 0],
                [0, 1, 1, 2, 1, 1, 4, 0, 0, 4],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 4, 0, 0, 0, 4],
            ]

        if name == "test_probabilistic_measurement":
            if "GPU" in self.device:  # pragma: no cover
                return {0: 273, 1: 233, 2: 242, 3: 252}

            return {0: 271, 1: 239, 2: 242, 3: 248}

        if name == "test_unbalanced_probabilistic_measurement":
            if "GPU" in self.device:  # pragma: no cover
                return {0: 196, 1: 153, 2: 156, 3: 495}

            return {0: 168, 1: 188, 2: 154, 3: 490}

        return [
            {5: 30},
            {5: 12, 7: 6, 4: 6, 1: 5, 6: 1},
            {3: 7, 6: 4, 2: 4, 7: 4, 0: 4, 5: 3, 4: 2, 1: 2},
        ]
