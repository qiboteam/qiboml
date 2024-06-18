import pytest

from qiboml.backends.jax import JaxBackend
from qiboml.backends.pytorch import PyTorchBackend
from qiboml.backends.tensorflow import TensorflowBackend

# backends to be tested
BACKENDS = [
    "tensorflow",
    "pytorch",
    "jax",
]


NAME2BACKEND = {
    "tensorflow": TensorflowBackend(),
    "pytorch": PyTorchBackend(),
    "jax": JaxBackend(),
}


@pytest.fixture
def backend(backend_name):
    yield NAME2BACKEND[backend_name]


def pytest_generate_tests(metafunc):
    module_name = metafunc.module.__name__

    if "backend_name" in metafunc.fixturenames:
        metafunc.parametrize("backend_name", BACKENDS)
