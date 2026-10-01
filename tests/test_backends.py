import platform

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
    tensorflow = False if platform.system() == "Windows" else True
    available_backends = {
        "tensorflow": tensorflow,
        "pytorch": True,
        "jax": True,
    }
    assert MetaBackend().list_available() == available_backends


@pytest.mark.parametrize("mode", ["full", "same", "valid"])
@pytest.mark.parametrize("sizes", [(6, 3), (3, 6), (5, 5), (5, 4), (4, 1)])
def test_convolve(backend, sizes, mode):
    backend.set_seed(10)
    array_1 = _random_complex(sizes[0], backend)
    array_2 = _random_complex(sizes[1], backend)

    target = np.convolve(
        backend.to_numpy(array_1), backend.to_numpy(array_2), mode=mode
    )
    result = backend.convolve(array_1, array_2, mode=mode)

    backend.assert_allclose(
        result, backend.cast(target, dtype=result.dtype), atol=1e-10
    )


def test_convolve_errors(backend):
    if backend.platform == "jax":
        pytest.skip("``jax.numpy.convolve`` is used directly.")

    backend.set_seed(10)
    array = _random_complex(3, backend)

    with pytest.raises(ValueError):
        backend.convolve(array, array, mode="wrong")
    with pytest.raises(ValueError):
        backend.convolve(array, array, stride=2)


@pytest.mark.parametrize("mode", ["full", "same", "valid"])
def test_convolve_real(backend, mode):
    backend.set_seed(10)
    array_1 = backend.random_normal(0.0, 1.0, size=6, dtype=backend.float64)
    array_2 = backend.random_normal(0.0, 1.0, size=3, dtype=backend.float64)

    target = np.convolve(
        backend.to_numpy(array_1), backend.to_numpy(array_2), mode=mode
    )
    result = backend.convolve(array_1, array_2, mode=mode)

    backend.assert_allclose(
        result, backend.cast(target, dtype=result.dtype), atol=1e-10
    )


@pytest.mark.parametrize("size", [4, 7])
def test_fft_ifft(backend, size):
    backend.set_seed(10)
    array = _random_complex(size, backend)
    _array = backend.to_numpy(array)

    result = backend.fft(array)
    backend.assert_allclose(
        result, backend.cast(np.fft.fft(_array), dtype=result.dtype), atol=1e-10
    )

    result = backend.ifft(array)
    backend.assert_allclose(
        result, backend.cast(np.fft.ifft(_array), dtype=result.dtype), atol=1e-10
    )


def test_fft_ifft_real(backend):
    backend.set_seed(10)
    array = backend.random_normal(0.0, 1.0, size=5, dtype=backend.float64)
    _array = backend.to_numpy(array)

    result = backend.fft(array)
    backend.assert_allclose(
        result, backend.cast(np.fft.fft(_array), dtype=result.dtype), atol=1e-10
    )

    result = backend.ifft(array)
    backend.assert_allclose(
        result, backend.cast(np.fft.ifft(_array), dtype=result.dtype), atol=1e-10
    )


@pytest.mark.parametrize("size", [2, 3])
def test_poly_matrix(backend, size):
    backend.set_seed(10)
    matrix = _random_complex((size, size), backend)

    target = np.poly(backend.to_numpy(matrix))
    result = backend.poly(matrix)

    backend.assert_allclose(
        result, backend.cast(target, dtype=result.dtype), atol=1e-10
    )


@pytest.mark.parametrize("degree", [1, 2, 5])
def test_poly_roots(backend, degree):
    backend.set_seed(10)
    roots = _random_complex(degree, backend)

    target = np.poly(backend.to_numpy(roots))
    coefficients = backend.poly(roots)
    backend.assert_allclose(
        coefficients, backend.cast(target, dtype=coefficients.dtype), atol=1e-10
    )

    # the roots of the (monic) polynomial give back its coefficients, whatever their
    # order, and the companion matrix of random complex roots is not symmetric
    backend.assert_allclose(
        backend.poly(backend.roots(coefficients)), coefficients, atol=1e-10
    )


def test_roots_edge_cases(backend):
    # 2 * x^2 - 3 * x + 1, with two zeros in the highest degrees
    leading = backend.cast([0.0, 0.0, 2.0, -3.0, 1.0], dtype=backend.float64)
    coefficients = backend.poly(backend.roots(leading))
    backend.assert_allclose(
        coefficients,
        backend.cast([1.0, -1.5, 0.5], dtype=coefficients.dtype),
        atol=1e-10,
    )

    # degree one
    linear = backend.cast([2.0, -6.0], dtype=backend.float64)
    result = backend.roots(linear)
    backend.assert_allclose(result, backend.cast([3.0], dtype=result.dtype))

    # no roots
    constant = backend.cast([5.0], dtype=backend.float64)
    assert len(backend.roots(constant)) == 0


def _random_complex(size, backend):
    """Complex array of normally distributed numbers with the given shape."""
    real = backend.random_normal(0.0, 1.0, size=size, dtype=backend.float64)
    imag = backend.random_normal(0.0, 1.0, size=size, dtype=backend.float64)

    return backend.cast(real, dtype=backend.complex128) + 1j * backend.cast(
        imag, dtype=backend.complex128
    )
