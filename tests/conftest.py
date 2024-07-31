"""conftest.py.

Pytest fixtures.
"""

import sys

import pytest
from qibo.backends import construct_backend

# backends to be tested
BACKENDS = [
    "numpy",
    "tensorflow",
    "pytorch",
    # "jax",
]

FRONTENDS = [
    "pytorch",
    "keras",
]


def get_backend(backend_name):
    if "-" in backend_name:
        name, platform = backend_name.split("-")
    else:
        name, platform = backend_name, None
    return construct_backend(name, platform=platform)


def get_frontend(frontend_name):
    import qiboml

    frontend = getattr(qiboml, frontend_name)
    setattr(frontend, "__str__", frontend_name)
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
