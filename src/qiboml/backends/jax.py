from qibo import __version__
from qibo.backends import einsum_utils
from qibo.backends.npmatrices import NumpyMatrices
from qibo.backends.numpy import NumpyBackend
from qibo.config import raise_error


class JaxMatrices(NumpyMatrices):
    def __init__(self, dtype):
        super().__init__(dtype)
        import jax  # pylint: disable=import-error
        import jax.numpy as jnp  # pylint: disable=import-error

        self.jax = jax
        self.np = jnp

    def _cast(self, x, dtype):
        return self.np.array(x, dtype=dtype)

    def Unitary(self, u):
        return self._cast(u, dtype=self.dtype)


class JaxBackend(NumpyBackend):
    def __init__(self):
        super().__init__()
        self.name = "jax"

        import jax
        import jax.numpy as jnp  # pylint: disable=import-error
        import numpy as np

        jax.config.update("jax_enable_x64", True)

        self.jax = jax
        self.numpy = np

        self.np = jnp

        self.versions = {
            "qibo": __version__,
            "numpy": np.__version__,
            "tensorflow": jax.__version__,
        }

        self.matrices = JaxMatrices(self.dtype)
        self.tensor_types = (jnp.ndarray, np.ndarray)

    def set_precision(self, precision):
        if precision != self.precision:
            if precision == "single":
                self.precision = precision
                self.dtype = self.np.complex64
            elif precision == "double":
                self.precision = precision
                self.dtype = self.np.complex128
            else:
                raise_error(ValueError, f"Unknown precision {precision}.")
            if self.matrices:
                self.matrices = self.matrices.__class__(self.dtype)

    def cast(self, x, dtype=None, copy=False):
        if dtype is None:
            dtype = self.dtype
        if isinstance(x, self.tensor_types):
            return x.astype(dtype)
        elif self.issparse(x):
            return x.astype(dtype)
        return self.np.array(x, dtype=dtype, copy=copy)

    # TODO: using numpy's rng for now. Shall we use Jax's?
    def set_seed(self, seed):
        self.numpy.random.seed(seed)

    def sample_shots(self, probabilities, nshots):
        return self.numpy.random.choice(
            range(len(probabilities)), size=nshots, p=probabilities
        )

    def zero_state(self, nqubits):
        state = self.np.zeros(2**nqubits, dtype=self.dtype)
        state = state.at[0].set(1)
        return state

    def zero_density_matrix(self, nqubits):
        state = self.np.zeros(2 * (2**nqubits,), dtype=self.dtype)
        state = state.at[0, 0].set(1)
        return state

    def plus_state(self, nqubits):
        state = self.np.ones(2**nqubits, dtype=self.dtype)
        state /= self.np.sqrt(2**nqubits)
        return state

    def plus_density_matrix(self, nqubits):
        state = self.np.ones(2 * (2**nqubits,), dtype=self.dtype)
        state /= 2**nqubits
        return state

    def matrix(self, gate):
        npmatrix = super().matrix(gate)
        return self.np.array(npmatrix, dtype=self.dtype)

    def matrix_parametrized(self, gate):
        npmatrix = super().matrix_parametrized(gate)
        return self.np.array(npmatrix, dtype=self.dtype)

    def matrix_fused(self, gate):
        npmatrix = super().matrix_fused(gate)
        return self.np.array(npmatrix, dtype=self.dtype)

    def update_frequencies(self, frequencies, probabilities, nsamples):
        samples = self.sample_shots(probabilities, nsamples)
        res, counts = self.np.unique(samples, return_counts=True)
        frequencies = frequencies.at[res].add(counts)
        return frequencies

    def apply_gate_density_matrix(self, gate, state, nqubits):
        state = self.cast(state)
        state = self.np.reshape(state, 2 * nqubits * (2,))
        matrix = gate.matrix(self)
        if gate.is_controlled_by:
            matrix = self.np.reshape(matrix, 2 * len(gate.target_qubits) * (2,))
            matrixc = self.np.conj(matrix)
            ncontrol = len(gate.control_qubits)
            nactive = nqubits - ncontrol
            n = 2**ncontrol

            order, targets = einsum_utils.control_order_density_matrix(gate, nqubits)
            state = self.np.transpose(state, order)
            state = self.np.reshape(state, 2 * (n,) + 2 * nactive * (2,))

            leftc, rightc = einsum_utils.apply_gate_density_matrix_controlled_string(
                targets, nactive
            )
            state01 = state[: n - 1, n - 1]
            state01 = self.np.einsum(rightc, state01, matrixc)
            state10 = state[n - 1, : n - 1]
            state10 = self.np.einsum(leftc, state10, matrix)

            left, right = einsum_utils.apply_gate_density_matrix_string(
                targets, nactive
            )
            state11 = state[n - 1, n - 1]
            state11 = self.np.einsum(right, state11, matrixc)
            state11 = self.np.einsum(left, state11, matrix)

            state00 = state[: n - 1]
            state00 = state00[:, tuple(range(n - 1))]
            state01 = self.np.concatenate([state00, state01[:, None]], axis=1)
            state10 = self.np.concatenate([state10, state11[None]], axis=0)
            state = self.np.concatenate([state01, state10[None]], axis=0)
            state = self.np.reshape(state, 2 * nqubits * (2,))
            state = self.np.transpose(state, einsum_utils.reverse_order(order))
        else:
            matrix = self.np.reshape(matrix, 2 * len(gate.qubits) * (2,))
            matrixc = self.np.conj(matrix)
            left, right = einsum_utils.apply_gate_density_matrix_string(
                gate.qubits, nqubits
            )
            state = self.np.einsum(right, state, matrixc)
            state = self.np.einsum(left, state, matrix)
        return self.np.reshape(state, 2 * (2**nqubits,))
