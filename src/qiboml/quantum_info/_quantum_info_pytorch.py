import torch  # pylint: disable=import-error

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


class QinfoTorch:
    pass


QINFO = QinfoTorch()

for function in (
    _unvectorization_column,
    _post_sparse_pauli_basis_vectorization,
):
    setattr(QINFO, function.__name__, function)
