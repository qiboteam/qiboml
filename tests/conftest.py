"""conftest.py.

Pytest fixtures.
"""

import sys

import pytest

# backends to be tested
BACKENDS = [
    "tensorflow",
    # "pytorch",
    # "jax",
]

FRONTENDS = [
    # "pytorch",
    "keras",
]


def get_backend(backend_name):
    from qibo.backends.pytorch import (  # the qiboml pytorch is not updated
        PyTorchBackend,
    )

    from qiboml.backends.jax import JaxBackend
    from qiboml.backends.tensorflow import TensorflowBackend

    NAME2BACKEND = {
        "tensorflow": TensorflowBackend,
        "pytorch": PyTorchBackend,
        "jax": JaxBackend,
    }

    return NAME2BACKEND[backend_name]()


def get_frontend(frontend_name):
    from qiboml.models import keras, pytorch

    if frontend_name == "keras":
        frontend = keras
    elif frontend_name == "pytorch":
        frontend = pytorch
    else:
        raise RuntimeError(f"Unknown frontend {frontend_name}.")
    return frontend


AVAILABLE_BACKENDS = []
for backend_name in BACKENDS:
    try:
        _backend = get_backend(backend_name)
        AVAILABLE_BACKENDS.append(backend_name)
    except (ModuleNotFoundError, ImportError):
        pass

AVAILABLE_FRONTENDS = []
for frontend_name in FRONTENDS:
    try:
        _frontend = get_frontend(frontend_name)
        AVAILABLE_FRONTENDS.append(frontend_name)
    except (ModuleNotFoundError, ImportError):
        pass


def pytest_runtest_setup(item):
    ALL = {"darwin", "linux"}
    supported_platforms = ALL.intersection(mark.name for mark in item.iter_markers())
    plat = sys.platform
    if supported_platforms and plat not in supported_platforms:  # pragma: no cover
        # case not covered by workflows
        pytest.skip(f"Cannot run test on platform {plat}.")


@pytest.fixture
def backend(backend_name):
    yield get_backend(backend_name)


@pytest.fixture
def frontend(frontend_name):
    yield get_frontend(frontend_name)


def pytest_runtest_setup(item):
    ALL = {"darwin", "linux"}
    supported_platforms = ALL.intersection(mark.name for mark in item.iter_markers())
    plat = sys.platform
    if supported_platforms and plat not in supported_platforms:  # pragma: no cover
        # case not covered by workflows
        pytest.skip(f"Cannot run test on platform {plat}.")


def pytest_configure(config):
    config.addinivalue_line("markers", "linux: mark test to run only on linux")


def pytest_generate_tests(metafunc):
    module_name = metafunc.module.__name__

    if "backend_name" in metafunc.fixturenames:
        metafunc.parametrize("backend_name", AVAILABLE_BACKENDS)
    if "frontend_name" in metafunc.fixturenames:
        metafunc.parametrize("frontend_name", AVAILABLE_FRONTENDS)
