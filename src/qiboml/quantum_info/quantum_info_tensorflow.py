import numpy as np
import tensorflow as tf
import tensorflow.experimental.numpy as tnp

ENGINE = tnp


def _masked_assignement(tensor, mask, values):
    return tf.tensor_scatter_nd_update(tensor, tf.where(mask), values)


def _sample_from_quantum_mallows_distribution(nqubits: int):
    exponents = ENGINE.arange(nqubits, 0, -1, dtype=ENGINE.int64).numpy()
    powers = 4**exponents
    powers[powers == 0] = np.iinfo(np.int64).max
    r = ENGINE.random.uniform(0, 1, size=(nqubits,)).numpy()
    indexes = (-1) * np.ceil(np.log2(r + (1 - r) / powers)).astype(np.int64)
    idx_le_exp = indexes < exponents
    hadamards = idx_le_exp.astype(np.int64)
    idx_gt_exp = idx_le_exp ^ True
    indexes[idx_gt_exp] = 2 * exponents[idx_gt_exp] - indexes[idx_gt_exp] - 1
    mute_index = list(range(nqubits))
    permutations = np.zeros(nqubits, dtype=int)
    for l, index in enumerate(indexes.tolist()):
        permutations[l] = mute_index[index]
        del mute_index[index]
    return hadamards, permutations


def _gamma_delta_matrices(nqubits: int, hadamards, permutations):
    delta_matrix = np.eye(nqubits, dtype=int)
    delta_matrix_prime = np.copy(delta_matrix)

    gamma_matrix_prime = ENGINE.random.randint(0, 2, size=nqubits).numpy()
    gamma_matrix_prime = np.diag(gamma_matrix_prime)

    gamma_matrix = ENGINE.random.randint(0, 2, size=nqubits).numpy()
    gamma_matrix = hadamards * gamma_matrix
    gamma_matrix = np.diag(gamma_matrix)

    tril_indices = np.tril_indices(nqubits, k=-1)
    delta_matrix_prime[tril_indices] = ENGINE.random.randint(
        0, 2, size=len(tril_indices[0])
    ).numpy()
    gamma_matrix_prime[tril_indices] = ENGINE.random.randint(
        0, 2, size=len(tril_indices[0])
    ).numpy()
    triu_indices = np.triu_indices(nqubits, k=1)
    gamma_matrix_prime[triu_indices] = gamma_matrix_prime[tril_indices]

    p_col_gt_row = permutations[triu_indices[1]] > permutations[triu_indices[0]]
    p_col_neq_row = permutations[triu_indices[1]] != permutations[triu_indices[0]]
    p_col_le_row = p_col_gt_row ^ True
    h_row_eq_0 = hadamards[triu_indices[0]] == 0
    h_col_eq_0 = hadamards[triu_indices[1]] == 0

    idx = (h_row_eq_0 * h_col_eq_0 ^ True) * p_col_neq_row
    elements = ENGINE.random.randint(0, 2, size=len(idx.nonzero()[0])).numpy()
    gamma_matrix[triu_indices[0][idx], triu_indices[1][idx]] = elements
    gamma_matrix[triu_indices[1][idx], triu_indices[0][idx]] = elements

    idx = p_col_gt_row | (p_col_le_row * h_row_eq_0 * h_col_eq_0)
    elements = ENGINE.random.randint(0, 2, size=len(idx.nonzero()[0])).numpy()
    delta_matrix[triu_indices[1][idx], triu_indices[0][idx]] = elements

    return gamma_matrix, gamma_matrix_prime, delta_matrix, delta_matrix_prime


class QinfoTensorflow:
    pass


QINFO = QinfoTensorflow()

for function in (_sample_from_quantum_mallows_distribution, _gamma_delta_matrices):
    setattr(QINFO, function.__name__, function)
