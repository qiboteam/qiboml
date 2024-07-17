import platform

import pytest

from qiboml.backends import MetaBackend


def test_metabackend_load(backend):
    assert isinstance(MetaBackend.load(backend.name), backend.__class__)


def test_metabackend_load_error():
    with pytest.raises(ValueError):
        MetaBackend.load("nonexistent-backend")


def test_metabackend_list_available():
    tensorflow = False if platform.system() == "Windows" else True
    available_backends = {"tensorflow": tensorflow, "pytorch": True, "jax": True}
    assert MetaBackend().list_available() == available_backends


def test_gradients_pytorch():
    from qiboml.backends import PyTorchBackend  # pylint: disable=import-outside-toplevel
    from qibo import gates # pylint: disable=import-outside-toplevel
    
    backend = PyTorchBackend()
    gate = gates.RX(0, 0.1)
    matrix = gate.matrix(backend)
    assert matrix.requires_grad
    assert backend.gradients
    backend.requires_grad(False)
    gate = gates.RX(0, 0.1)
    matrix = gate.matrix(backend)
    assert not matrix.requires_grad
    assert not backend.gradients
    assert not backend.matrices.requires_grad
