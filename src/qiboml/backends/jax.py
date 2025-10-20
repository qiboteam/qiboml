"""Module defining the Jax backend."""

from functools import partial
from typing import Optional, Union, Tuple

import jax  # pylint: disable=import-error
import jax.numpy as jnp  # pylint: disable=import-error
import numpy as np
from qibo import __version__
from qibo.backends import Backend, einsum_utils
from qibo.backends.npmatrices import NumpyMatrices
from qibo.result import CircuitResult, QuantumState


@partial(jax.jit, static_argnums=(0, 1, 2))
def zero_state(nqubits, density_matrix, dtype):
    if density_matrix:
        state = jnp.zeros(2 * (2**nqubits,), dtype=dtype).at[0, 0].set(1)
    else:
        state = jnp.zeros(2**nqubits, dtype=dtype).at[0].set(1)
    return state


@partial(jax.jit, static_argnames={"dtype"})
def cast_matrix(x, dtype):
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


class JaxMatrices(NumpyMatrices):

    def __init__(self, dtype):
        super().__init__(dtype)
        self.np = jnp
        self.dtype = dtype

    def _cast(self, x, dtype):
        return cast_matrix(x, dtype)


class JaxBackend(Backend):
    def __init__(self):
        super().__init__()

        self.engine = jnp
        self.jax = jax
        self.jax.config.update("jax_enable_x64", True)

        self.dtype = self.complex128
        self.matrices = JaxMatrices(self.dtype)
        self.name = "qiboml"
        self.numeric_types += (
            self.int8,
            self.int32,
            self.int64,
            self.float32,
            self.float64,
            self.complex64,
            self.complex128,
        )
        self.platform = "jax"
        self.tensor_types = (self.engine.ndarray,)

    def cast(self, array, dtype=None, copy=False):
        if dtype is None:
            dtype = self.dtype

        if isinstance(array, self.tensor_types):
            return array.astype(dtype)

        if self.is_sparse(array):
            return array.astype(dtype)

        return self.engine.array(array, dtype=dtype, copy=copy)

    def to_numpy(self, array):

        if isinstance(array, (list, tuple)):
            return np.asarray([self.to_numpy(elem) for elem in array])

        return np.asarray(array)

    # TODO: using numpy's rng for now. Shall we use Jax's?
    def set_seed(self, seed):
        np.random.seed(seed)

    def default_rng(self, seed: Optional[int] = None) -> "ndarray":
        return np.random.default_rng(seed)

    def random_choice(
        self,
        array,
        size: Optional[Union[int, Tuple[int, ...]]] = None,
        replace: bool = True,
        p=None,
        seed=None,
    ) -> "ndarray":
        if seed is not None:
            local_state = self.default_rng(seed) if isinstance(seed, int) else seed

            return local_state.choice(array, size=size, replace=replace, p=p)

        return np.random.choice(array, size=size, replace=replace, p=p)

    def random_integers(
        self,
        low: int,
        high: Optional[int] = None,
        size: Optional[Union[int, Tuple[int, ...]]] = None,
        seed=None,
    ):
        if high is None:
            high = low
            low = 0

        if size is None:
            size = 1

        if seed is not None:
            local_state = self.default_rng(seed) if isinstance(seed, int) else seed

            return local_state.integers(low, high, size)

        return np.random.randint(low, high, size)

    def random_sample(self, size: int, seed=None):
        if seed is not None:
            local_state = self.default_rng(seed) if isinstance(seed, int) else seed

            return local_state.random(size)

        return np.random.random(size)

    def random_uniform(
        self,
        low: Union[float, int] = 0.0,
        high: Union[float, int] = 1.0,
        size: Optional[Union[int, Tuple[int, ...]]] = None,
        seed=None,
    ) -> "ndarray":
        if seed is not None:
            local_state = self.default_rng(seed) if isinstance(seed, int) else seed

            return local_state.uniform(low, high, size)

        return np.random.uniform(low, high, size)

    def matrix_fused(self, fgate):
        rank = len(fgate.target_qubits)
        # jax only supports coo sparse arrays
        # however they are probably not as efficient as csr ones
        # indeed using dense arrays instead of coo ones proved to be significantly faster
        matrix = self.identity(2**rank)

        for gate in fgate.gates:
            gmatrix = gate.matrix(self)
            # add controls if controls were instantiated using
            # the ``Gate.controlled_by`` method
            num_controls = len(gate.control_qubits)
            if num_controls > 0:
                gmatrix = self.jax.scipy.linalg.block_diag(
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
            matrix = gmatrix @ matrix

        return matrix

    def zero_state(self, nqubits: int, density_matrix: bool = False, dtype=None):
        if dtype is None:
            dtype = self.dtype

        return zero_state(nqubits, density_matrix, dtype)

    def update_frequencies(self, frequencies, probabilities, nsamples):
        samples = self.sample_shots(probabilities, nsamples)
        res, counts = self.unique(samples, return_counts=True)
        frequencies = frequencies.at[res].add(counts)
        return frequencies

    def matrix(self, gate):
        matrix = super().matrix(gate)
        if isinstance(matrix, self.jax.core.Tracer):
            delattr(self.matrices, gate.__class__.__name__)
        return matrix

    def apply_gate(self, gate, state, nqubits):
        density_matrix = bool(len(state.shape) == 2)

        if density_matrix:
            return self._apply_gate_density_matrix(gate, state, nqubits)

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

    def assert_allclose(
        self, value, target, rtol: float = 1e-7, atol: float = 0.0
    ):  # pragma: no cover
        if isinstance(value, (CircuitResult, QuantumState)):
            value = value.state()
        if isinstance(target, (CircuitResult, QuantumState)):
            target = target.state()

        np.testing.assert_allclose(value, target, rtol=rtol, atol=atol)

    def _apply_gate_density_matrix(self, gate, state, nqubits: int):
        state = self.cast(state)
        state = self.reshape(state, 2 * nqubits * (2,))
        matrix = gate.matrix(self)
        if gate.is_controlled_by:
            matrix = self.reshape(matrix, 2 * len(gate.target_qubits) * (2,))
            matrixc = self.conj(matrix)
            ncontrol = len(gate.control_qubits)
            nactive = nqubits - ncontrol
            n = 2**ncontrol

            order, targets = einsum_utils.control_order_density_matrix(gate, nqubits)
            state = self.transpose(state, order)
            state = self.reshape(state, 2 * (n,) + 2 * nactive * (2,))

            leftc, rightc = einsum_utils.apply_gate_density_matrix_controlled_string(
                targets, nactive
            )
            state01 = state[: n - 1, n - 1]
            state01 = self.einsum(rightc, state01, matrixc)
            state10 = state[n - 1, : n - 1]
            state10 = self.einsum(leftc, state10, matrix)

            left, right = einsum_utils.apply_gate_density_matrix_string(
                targets, nactive
            )
            state11 = state[n - 1, n - 1]
            state11 = self.einsum(right, state11, matrixc)
            state11 = self.einsum(left, state11, matrix)

            state00 = state[: n - 1]
            state00 = state00[:, tuple(range(n - 1))]
            state01 = self.concatenate([state00, state01[:, None]], axis=1)
            state10 = self.concatenate([state10, state11[None]], axis=0)
            state = self.concatenate([state01, state10[None]], axis=0)
            state = self.reshape(state, 2 * nqubits * (2,))
            state = self.transpose(state, einsum_utils.reverse_order(order))
        else:
            matrix = self.reshape(matrix, 2 * len(gate.qubits) * (2,))
            matrixc = self.conj(matrix)
            left, right = einsum_utils.apply_gate_density_matrix_string(
                gate.qubits, nqubits
            )
            state = self.einsum(right, state, matrixc)
            state = self.einsum(left, state, matrix)
        return self.reshape(state, 2 * (2**nqubits,))
