import sys

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


def test_tensorflow_set_threads(backend, caplog):
    if backend.platform != "tensorflow":
        pytest.skip("Only the tensorflow backend warns on ``set_threads``.")

    backend.set_threads(backend.nthreads)
    assert "set_threads" not in caplog.text

    backend.set_threads(backend.nthreads + 1)
    assert "set_threads" in caplog.text
