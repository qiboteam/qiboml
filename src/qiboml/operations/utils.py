from inspect import signature
from typing import Callable

import numpy as np
from qibo.backends import Backend

from qiboml import ndarray


def circuit_trace(
    f: Callable, backend: Backend, jacobian: Callable
) -> tuple[dict[int, tuple[int]], ndarray]:
    nparams = len(signature(f).parameters)
    params = backend.cast(
        np.random.randn(nparams), dtype=backend.np.float64, device=backend.device
    )

    def build(x):
        # one parameter gates only
        return tuple(par[0] for par in f(backend.np, *x).get_parameters())

    jac = jacobian(build, params)
    par_map = {}
    for i, row in enumerate(jac):
        for j in backend.np.nonzero(row):
            j = int(j)
            if j in par_map:
                par_map[j] += (i,)
            else:
                par_map[j] = (i,)
    return par_map, params
