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


class QinfoJax:
    pass


QINFO = QinfoJax()

for function in (_sample_from_quantum_mallows_distribution,):
    setattr(QINFO, function.__name__, function)
