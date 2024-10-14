import pytest

from qiboml.backends.jax import JaxBackend
from qiboml.backends.pytorch import PyTorchBackend
from qiboml.backends.tensorflow import TensorflowBackend

# backends to be tested
BACKENDS = [
    "tensorflow",
]


NAME2BACKEND = {
    "tensorflow": TensorflowBackend,
}


@pytest.fixture
def backend(backend_name):
    yield NAME2BACKEND[backend_name]()


AVAILABLE_BACKENDS = []
for backend_name in BACKENDS:
    try:
        _backend = NAME2BACKEND[backend_name]()
        AVAILABLE_BACKENDS.append(backend_name)
    except (ModuleNotFoundError, ImportError):
        pass


def pytest_generate_tests(metafunc):
    module_name = metafunc.module.__name__

    if "backend_name" in metafunc.fixturenames:
        metafunc.parametrize("backend_name", AVAILABLE_BACKENDS)
