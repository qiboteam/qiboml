from functools import partial

import jax
import jax.numpy as jnp  # pylint: disable=import-error
import numpy as np
from jax.experimental import sparse
from qibo import __version__
from qibo.backends import einsum_utils
from qibo.backends.npmatrices import NumpyMatrices
from qibo.backends.numpy import NumpyBackend
from qibo.config import raise_error


class JaxMatrices(NumpyMatrices):

    def __init__(self, dtype):
        super().__init__(dtype)
        self.np = jnp
        self.dtype = dtype

    def _cast(self, x, dtype):
        return jnp.asarray(x, dtype=dtype)


@partial(jax.jit, static_argnums=(2, 3))
def _apply_gate(matrix, state, qubits, nqubits):
    state = jnp.reshape(state, nqubits * (2,))
    matrix = jnp.reshape(matrix, 2 * len(qubits) * (2,))
    opstring = einsum_utils.apply_gate_string(qubits, nqubits)
    state = jnp.einsum(opstring, state, matrix)
    return jnp.reshape(state, (2**nqubits,))


@partial(jax.jit, static_argnums=(4, 5, 6))
def _apply_gate_controlled(
    matrix, state, order, targets, control_qubits, target_qubits, nqubits
):
    state = jnp.reshape(state, nqubits * (2,))
    matrix = jnp.reshape(matrix, 2 * len(target_qubits) * (2,))
    ncontrol = len(control_qubits)
    nactive = nqubits - ncontrol
    state = jnp.transpose(state, order)
    # Apply `einsum` only to the part of the state where all controls
    # are active. This should be `state[-1]`
    state = jnp.reshape(state, (2**ncontrol,) + nactive * (2,))
    opstring = einsum_utils.apply_gate_string(targets, nactive)
    updates = jnp.einsum(opstring, state[-1], matrix)
    # Concatenate the updated part of the state `updates` with the
    # part of of the state that remained unaffected `state[:-1]`.
    state = jnp.concatenate([state[:-1], updates[None]], axis=0)
    state = jnp.reshape(state, nqubits * (2,))
    # Put qubit indices back to their proper places
    state = jnp.transpose(state, einsum_utils.reverse_order(order))
    return jnp.reshape(state, (2**nqubits,))


class JaxBackend(NumpyBackend):
    def __init__(self):
        super().__init__()

        self.name = "qiboml"
        self.platform = "jax"

        import jax
        import jax.numpy as jnp  # pylint: disable=import-error
        import numpy

        jax.config.update("jax_enable_x64", True)

        self.jax = jax
        self.numpy = numpy

        self.np = jnp
        self.tensor_types = (jnp.ndarray, numpy.ndarray)
        self.matrices = JaxMatrices(self.dtype)

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
        elif self.is_sparse(x):
            return x.astype(dtype)
        return self.np.array(x, dtype=dtype, copy=copy)

    def to_numpy(self, x):

        if isinstance(x, list) or isinstance(x, tuple):
            return self.numpy.asarray([self.to_numpy(i) for i in x])

        return self.numpy.asarray(x)

    # TODO: using numpy's rng for now. Shall we use Jax's?
    def set_seed(self, seed):
        self.numpy.random.seed(seed)

    def sample_shots(self, probabilities, nshots):
        return self.numpy.random.choice(
            range(len(probabilities)), size=nshots, p=probabilities
        )

    def matrix_fused(self, fgate):
        rank = len(fgate.target_qubits)
        # jax only supports coo sparse arrays
        # however they are probably not as efficient as csr ones
        # indeed using dense arrays instead of coo ones proved to be significantly faster
        matrix = self.np.eye(2**rank)

        for gate in fgate.gates:
            gmatrix = gate.matrix(self)
            # add controls if controls were instantiated using
            # the ``Gate.controlled_by`` method
            num_controls = len(gate.control_qubits)
            if num_controls > 0:
                gmatrix = self.jax.scipy.linalg.block_diag(
                    self.np.eye(2 ** len(gate.qubits) - len(gmatrix)), gmatrix
                )
            # Kronecker product with identity is needed to make the
            # original matrix have shape (2**rank x 2**rank)
            eye = self.np.eye(2 ** (rank - len(gate.qubits)))
            gmatrix = self.np.kron(gmatrix, eye)
            # Transpose the new matrix indices so that it targets the
            # target qubits of the original gate
            original_shape = gmatrix.shape
            gmatrix = self.np.reshape(gmatrix, 2 * rank * (2,))
            qubits = list(gate.qubits)
            indices = qubits + [q for q in fgate.target_qubits if q not in qubits]
            indices = np.argsort(indices)
            transpose_indices = list(indices)
            transpose_indices.extend(indices + rank)
            gmatrix = self.np.transpose(gmatrix, transpose_indices)
            gmatrix = self.np.reshape(gmatrix, original_shape)
            matrix = gmatrix @ matrix

        return matrix

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

    def update_frequencies(self, frequencies, probabilities, nsamples):
        samples = self.sample_shots(probabilities, nsamples)
        res, counts = self.np.unique(samples, return_counts=True)
        frequencies = frequencies.at[res].add(counts)
        return frequencies

    def matrix(self, gate):
        matrix = super().matrix(gate)
        if isinstance(matrix, self.jax.core.Tracer):
            delattr(self.matrices, gate.__class__.__name__)
        return matrix

    def apply_gate(self, gate, state, nqubits):
        if gate.is_controlled_by:
            order, targets = einsum_utils.control_order(gate, nqubits)
            return _apply_gate_controlled(
                gate.matrix(self),
                state,
                order,
                targets,
                gate.control_qubits,
                gate.target_qubits,
                nqubits,
            )
        return _apply_gate(gate.matrix(self), state, gate.qubits, nqubits)

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
