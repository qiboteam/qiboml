import sys

import numpy as np
import pytest

from qiboml.backends import MetaBackend


def test_metabackend_load(backend):
    name = backend.name if backend.name != "qiboml" else backend.platform
    assert isinstance(MetaBackend.load(name), backend.__class__)


def test_metabackend_load_error():
    with pytest.raises(ValueError):
        MetaBackend.load("nonexistent-backend")


def test_metabackend_list_available():
    tensorflow = bool(sys.version_info < (3, 14))
    available_backends = {
        "tensorflow": tensorflow,
        "pytorch": True,
        "jax": True,
    }
    assert MetaBackend().list_available() == available_backends


def test_tensorflow_set_threads(backend, monkeypatch):
    if backend.platform != "tensorflow":
        pytest.skip(
            "Only the tensorflow backend relies on tensorflow's thread setters."
        )

    # same number of threads is a no-op, as used by qibo's parallel executions
    backend.set_threads(backend.nthreads)

    # tensorflow rejects changes after its runtime is initialized
    backend.cast([1.0])
    with pytest.raises(RuntimeError):
        backend.set_threads(backend.nthreads + 1)

    calls = []
    threading = backend.engine.config.threading
    monkeypatch.setattr(
        threading, "set_inter_op_parallelism_threads", lambda n: calls.append(n)
    )
    monkeypatch.setattr(
        threading, "set_intra_op_parallelism_threads", lambda n: calls.append(n)
    )
    backend.set_threads(2)
    assert backend.nthreads == 2
    assert calls == [2, 2]


def test_to_numpy_requires_grad(backend):
    torch = pytest.importorskip("torch")
    tensor = torch.tensor([1.0, 2.0], dtype=torch.float64, requires_grad=True)
    backend.assert_allclose(backend.to_numpy(tensor), np.array([1.0, 2.0]))
