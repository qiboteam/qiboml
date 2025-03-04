import jax
import jax.numpy as jnp
import numpy as np

ENGINE = jnp


def _sample_from_quantum_mallows_distribution(nqubits: int):
    exponents = ENGINE.arange(nqubits, 0, -1, dtype=ENGINE.int64)
    powers = 4**exponents
    powers.at[powers == 0].set(ENGINE.iinfo(ENGINE.int64).max)
    r = ENGINE.random.uniform(0, 1, (nqubits,))
    indexes = (-1) * ENGINE.ceil(ENGINE.log2(r + (1 - r) / powers)).astype(ENGINE.int64)
    idx_le_exp = indexes < exponents
    hadamards = idx_le_exp.astype(ENGINE.int64)
    idx_gt_exp = idx_le_exp ^ True
    indexes.at[idx_gt_exp].set(2 * exponents[idx_gt_exp] - indexes[idx_gt_exp] - 1)
    mute_index = list(range(nqubits))
    permutations = ENGINE.zeros(nqubits, dtype=int)
    for l, index in enumerate(indexes.tolist()):
        permutations.at[l].set(mute_index[index])
        del mute_index[index]
    return hadamards, permutations


def _gamma_delta_matrices(nqubits: int, hadamards, permutations):
    delta_matrix = ENGINE.eye(nqubits, dtype=int)
    delta_matrix_prime = ENGINE.copy(delta_matrix)

    gamma_matrix_prime = ENGINE.random.randint(0, 2, size=nqubits)
    gamma_matrix_prime = ENGINE.diag(gamma_matrix_prime)

    gamma_matrix = ENGINE.random.randint(0, 2, size=nqubits)
    gamma_matrix = hadamards * gamma_matrix
    gamma_matrix = ENGINE.diag(gamma_matrix)

    tril_indices = ENGINE.tril_indices(nqubits, k=-1)
    delta_matrix_prime.at[tril_indices].set(
        ENGINE.random.randint(0, 2, size=len(tril_indices[0]))
    )
    gamma_matrix_prime.at[tril_indices].set(
        ENGINE.random.randint(0, 2, size=len(tril_indices[0]))
    )
    triu_indices = ENGINE.triu_indices(nqubits, k=1)
    gamma_matrix_prime.at[triu_indices].set(gamma_matrix_prime[tril_indices])

    p_col_gt_row = permutations[triu_indices[1]] > permutations[triu_indices[0]]
    p_col_neq_row = permutations[triu_indices[1]] != permutations[triu_indices[0]]
    p_col_le_row = p_col_gt_row ^ True
    h_row_eq_0 = hadamards[triu_indices[0]] == 0
    h_col_eq_0 = hadamards[triu_indices[1]] == 0

    idx = (h_row_eq_0 * h_col_eq_0 ^ True) * p_col_neq_row
    elements = ENGINE.random.randint(0, 2, size=len(idx.nonzero()[0]))
    gamma_matrix.at[triu_indices[0][idx], triu_indices[1][idx]].set(elements)
    gamma_matrix.at[triu_indices[1][idx], triu_indices[0][idx]].set(elements)

    idx = p_col_gt_row | (p_col_le_row * h_row_eq_0 * h_col_eq_0)
    elements = ENGINE.random.randint(0, 2, size=len(idx.nonzero()[0]))
    delta_matrix.at[triu_indices[1][idx], triu_indices[0][idx]].set(elements)

    return gamma_matrix, gamma_matrix_prime, delta_matrix, delta_matrix_prime


class QinfoTensorflow:
    pass


QINFO = QinfoTensorflow()

for function in (_sample_from_quantum_mallows_distribution, _gamma_delta_matrices):
    setattr(QINFO, function.__name__, function)
