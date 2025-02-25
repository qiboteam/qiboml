import torch

ENGINE = torch


def _unvectorization_column(state: torch.Tensor, dim: int) -> torch.Tensor:
    return ENGINE.reshape(state.T, (dim, dim, state.shape[0])).T


def _post_sparse_pauli_basis_vectorization(
    basis: torch.Tensor, dim: int
) -> tuple[torch.Tensor, torch.Tensor]:
    indices = ENGINE.nonzero(basis, as_tuple=True)
    basis = basis[indices].reshape(-1, dim)
    indices = indices[1].reshape(-1, dim)
    return basis, indices


def _gamma_delta_matrices(
    nqubits: int, hadamards: torch.Tensor, permutations: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    delta_matrix = ENGINE.eye(nqubits, dtype=int)
    delta_matrix_prime = ENGINE.copy(delta_matrix)

    gamma_matrix_prime = ENGINE.randint(0, 2, size=(nqubits,))
    gamma_matrix_prime = ENGINE.diag(gamma_matrix_prime)

    gamma_matrix = ENGINE.randint(0, 2, size=(nqubits,))
    gamma_matrix = hadamards * gamma_matrix
    gamma_matrix = ENGINE.diag(gamma_matrix)

    tril_indices = ENGINE.tril_indices(nqubits, nqubits, offset=-1)
    if ENGINE.numel(tril_indices) > 0:
        delta_matrix_prime[tril_indices[0], tril_indices[1]] = ENGINE.randint(
            0, 2, size=(tril_indices.shape[-1],)
        )
        gamma_matrix_prime[tril_indices[0], tril_indices[1]] = ENGINE.randint(
            0, 2, size=(tril_indices.shape[-1],)
        )

    triu_indices = ENGINE.triu_indices(nqubits, nqubits, offset=1)
    if ENGINE.numel(triu_indices) > 0:
        gamma_matrix_prime[triu_indices[0], triu_indices[1]] = gamma_matrix_prime[
            tril_indices[0], tril_indices[1]
        ]

        p_col_gt_row = permutations[triu_indices[1]] > permutations[triu_indices[0]]
        p_col_neq_row = permutations[triu_indices[1]] != permutations[triu_indices[0]]
        p_col_le_row = p_col_gt_row ^ True
        h_row_eq_0 = hadamards[triu_indices[0]] == 0
        h_col_eq_0 = hadamards[triu_indices[1]] == 0

        idx = (h_row_eq_0 * h_col_eq_0 ^ True) * p_col_neq_row
        elements = ENGINE.randint(0, 2, size=(idx.nonzero().shape[0],))
        gamma_matrix[triu_indices[0][idx], triu_indices[1][idx]] = elements
        gamma_matrix[triu_indices[1][idx], triu_indices[0][idx]] = elements

        idx = p_col_gt_row | (p_col_le_row * h_row_eq_0 * h_col_eq_0)
        elements = ENGINE.randint(0, 2, size=(idx.nonzero().shape[0],))
        delta_matrix[triu_indices[1][idx], triu_indices[0][idx]] = elements

    return gamma_matrix, gamma_matrix_prime, delta_matrix, delta_matrix_prime


class QinfoTorch:
    pass


QINFO = QinfoTorch()

for function in (
    _unvectorization_column,
    _gamma_delta_matrices,
    _post_sparse_pauli_basis_vectorization,
):
    setattr(QINFO, function.__name__, function)
