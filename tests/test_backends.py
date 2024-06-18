import pytest

from qiboml.backends import MetaBackend


def test_metabackend_load(backend):
    assert isinstance(MetaBackend.load(backend.name), backend.__class__)


def test_metabackend_load_error():
    with pytest.raises(ValueError):
        MetaBackend.load("nonexistent-backend")


def test_metabackend_list_available():
    available_backends = {"tensorflow": True, "pytorch": True, "jax": True}
    assert MetaBackend().list_available() == available_backends
