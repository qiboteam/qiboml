import platform

import pytest

from qiboml.backends import MetaBackend


def test_metabackend_load(backend):
    name = backend.name if backend.name != "qiboml" else backend.platform
    assert isinstance(MetaBackend.load(name), backend.__class__)


def test_metabackend_load_error():
    with pytest.raises(ValueError):
        MetaBackend.load("nonexistent-backend")


def test_metabackend_list_available():
    tensorflow = False if platform.system() == "Windows" else True
    available_backends = {
        "tensorflow": tensorflow,
        "pytorch": True,
    }  # TODO: restore this --> , "pytorch": True, "jax": True}
    assert MetaBackend().list_available() == available_backends
