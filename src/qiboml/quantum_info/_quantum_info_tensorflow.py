import numpy as np
import tensorflow as tf  # pylint: disable=import-error
import tensorflow.experimental.numpy as tnp  # pylint: disable=import-error

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


class QinfoTensorflow:
    pass


QINFO = QinfoTensorflow()

for function in (_sample_from_quantum_mallows_distribution,):
    setattr(QINFO, function.__name__, function)
