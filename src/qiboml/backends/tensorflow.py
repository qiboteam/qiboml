"""Module defining the TensorFlow backend."""

import os
from collections import Counter
from typing import List, Optional, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike, DTypeLike
from qibo import Circuit, __version__
from qibo.backends import Backend
from qibo.backends.npmatrices import NumpyMatrices
from qibo.config import TF_LOG_LEVEL, log, raise_error
from qibo.gates.abstract import Gate
from qibo.result import CircuitResult, MeasurementOutcomes, QuantumState


class TensorflowMatrices(NumpyMatrices):
    # Redefine parametrized gate matrices for backpropagation to work

    def __init__(self, dtype):
        super().__init__(dtype)
        import tensorflow as tf  # pylint: disable=import-error,C0415

        self.engine = tf
        self.engine.conj = self.engine.math.conj

    def _cast(self, array, dtype) -> ArrayLike:
        return self.engine.cast(array, dtype=dtype)

    def Unitary(self, u: ArrayLike) -> ArrayLike:
        return self._cast(u, dtype=self.dtype)


class TensorflowBackend(Backend):
    def __init__(self):
        super().__init__()
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = str(TF_LOG_LEVEL)

        import tensorflow as tf  # pylint: disable=import-error,C0415
        from tensorflow.python.framework import (  # pylint: disable=E0611,C0415,import-error
            errors_impl,
        )

        self.engine = tf
        self.engine.experimental.numpy.experimental_enable_numpy_behavior()

        if TF_LOG_LEVEL >= 2:
            self.engine.get_logger().setLevel("ERROR")

        self.matrices = TensorflowMatrices(self.dtype)
        self.name = "qiboml"
        self.nthreads = 0
        self.oom_error = errors_impl.ResourceExhaustedError
        self.platform = "tensorflow"
        self.tensor_types = (np.ndarray, tf.Tensor, tf.Variable)
        self.versions = {
            "qibo": __version__,
            "numpy": np.__version__,
            "tensorflow": tf.__version__,
        }

        cpu_devices = self.engine.config.list_logical_devices("CPU")
        gpu_devices = self.engine.config.list_logical_devices("GPU")
        if gpu_devices:  # pragma: no cover
            # CI does not use GPUs
            self.device = gpu_devices[0].name
        elif cpu_devices:
            self.device = cpu_devices[0].name

    def cast(
        self, array: ArrayLike, dtype: Optional[DTypeLike] = None, copy: bool = False
    ) -> ArrayLike:
        """Cast an object as the array type of the current backend.

        Args:
            array (ArrayLike): Object to cast to array.
            dtype (str or type, optional): data type of ``array`` after casting.
                Options are ``"complex128"``, ``"complex64"``, ``"float64"``,
                or ``"float32"``. If ``None``, defaults to ``Backend.dtype``.
                Defaults to ``None``.
            copy (bool, optional): If ``True`` a copy of the object is created in memory.
                Defaults to ``False``.

        Returns:
            ArrayLike: ``array`` casted to ``dtype``, possibly copied in memory.
        """
        if dtype is None:
            dtype = self.dtype

        array = self.engine.cast(array, dtype=dtype)

        if copy:
            return self.copy(array)

        return array

    def compile(self, func: callable):
        return self.engine.function(func)

    def is_sparse(self, array: ArrayLike) -> bool:
        """Determine if a given array is a sparse tensor.

        Args:
            array (ArrayLike): array to determine the sparsity of.

        Returns:
            bool: ``True`` if ``array`` is sparse, ``False`` otherwise.
        """
        return isinstance(array, self.engine.sparse.SparseTensor)

    def set_device(self, device: str) -> None:  # pragma: no cover
        """Set simulation device. Works in-place.

        Args:
            device (str): Device index, *e.g.* ``/CPU:0`` for CPU, or ``/GPU:1`` for
                the second GPU in a multi-GPU environment.
        """
        self.device = device

    def set_seed(self, seed: Union[int, None]) -> None:
        """Set the seed of the random number generator. Works in-place.

        Args:
            seed (int or None): seed to be set. If ``None``, seed is random.
        """
        self.engine.random.set_seed(seed)

    def set_threads(self, nthreads: int) -> None:
        """Set number of threads for CPU backend simulations that accept it. Works in-place.

        Args:
            nthreads (int): Number of threads.
        """
        log.warning(
            "`set_threads` is not supported by the tensorflow "
            "backend. Please use tensorflow's thread setters: "
            "`tf.config.threading.set_inter_op_parallelism_threads` "
            "or `tf.config.threading.set_intra_op_parallelism_threads` "
            "to switch the number of threads."
        )

    def to_numpy(self, array: ArrayLike) -> ArrayLike:
        """Convert ``array`` to a ``numpy.ndarray``.

        Args:
            array (ArrayLike): array to be converted to ``numpy.ndarray``.

        Returns:
            ArrayLike: Original array converted to ``numpy.ndarray``.
        """
        return np.array(array)

    ########################################################################################
    ######## Methods related to array manipulation                                  ########
    ########################################################################################

    def all(self, array: ArrayLike, **kwargs) -> Union[ArrayLike, bool]:
        """Test whether all ``array`` elements evaluate to ``True``.

        Args:
            array (ArrayLike): array to be evaluated.
            kwargs (optional): additional options for this function.
                For more details, see the corresponding engine's documentation.

        Returns:
            bool or ArrayLike: If no axis is specified, returns ``True`` if all elements evaluate
            to ``True``, ``False`` otherwise. If an axis is specified, returns element-wise ``True``
            or ``False`` along the specified axis.
        """
        return self.engine.reduce_all(array, **kwargs)

    def arccos(self, array: ArrayLike, **kwargs) -> ArrayLike:
        """Calculate the element-wise inverse cosine of values in ``array``.

        Args:
            array (ArrayLike): array to calculate ``arccos`` of.
            kwargs (optional): additional options for this function.
                For more details, see the corresponding engine's documentation.

        Returns:
            ArrayLike: Array with the element-wise inverse cosine of elements in `array``.
        """
        return self.engine.math.acos(array, **kwargs)

    def arcsin(self, array: ArrayLike, **kwargs) -> ArrayLike:
        """Calculate the element-wise inverse sine of values in ``array``.

        Args:
            array (ArrayLike): array to calculate ``arcsin`` of.
            kwargs (optional): additional options for this function.
                For more details, see the corresponding engine's documentation.

        Returns:
            ArrayLike: Array with the element-wise inverse sine of elements in `array``.
        """
        return self.engine.math.asin(array, **kwargs)

    def arctan2(self, array_1: ArrayLike, array_2: ArrayLike, **kwargs) -> ArrayLike:
        """Return the element-wise principal argument of a complex number.

        For a complex number :math:`x + i \\, y`, the :math:`\\textrm{arctan2}` function is defined
        as

        .. math::
            \\textrm{arctan2}(y, \\, x) = \\textrm{arg}(x + i \\, y) =
                \\textrm{Im}(\\log(x + i\\,y)) \\, ,

        where :math:`\\textrm{Im}` indicates the imaginary part.

        Args:
            array_1 (ArrayLike): array of :math:`y` elements.
            array_2 (ArrayLike): array of :math:`x` elements.
            kwargs (optional): additional options for this function.
                For more details, see the corresponding engine's documentation.

        Returns:
            ArrayLike: Array of angles in radians, in the range :math:`[-\\pi, \\, \\pi]`.
        """
        return self.engine.math.atan2(array_1, array_2, **kwargs)

    def concatenate(self, tup: Tuple[ArrayLike, ...], **kwargs) -> ArrayLike:
        """Join a sequence of arrays along an existing axis.

        Args:
            arrays (Tuple[ArrayLike, ...]): tuple of arrays to be concatenated.
            kwargs (optional): additional options for this function.
                For more details, see the corresponding engine's documentation.

        Returns:
            ArrayLike: The concatenated array.
        """
        return self.engine.concat(tup, **kwargs)

    def conj(self, array: ArrayLike) -> ArrayLike:
        """Return the element-wise complex conjugate of ``array``.

        Args:
            array (ArrayLike): input array.

        Returns:
            ArrayLike: The complex conjugate of ``array``.
        """
        return self.engine.math.conj(array)

    def coo_matrix(self, array: ArrayLike, **kwargs) -> ArrayLike:  # pragma: no cover
        """Return the sparse version of ``array`` in coordinate format.

        Also known as the ``ijv`` or ``triplet`` format.

        Args:
            array (ArrayLike): input array.
            kwargs (optional): additional options for this function.
                For more details, see the corresponding engine's documentation.

        Returns:
            ArrayLike: The coordinate-format version of ``array``.
        """
        return self.engine.sparse.from_dense(array, **kwargs)

    def copy(self, array: ArrayLike, **kwargs) -> ArrayLike:
        """Return a copy of ``array``.

        Args:
            array (ArrayLike): input array.
            kwargs (optional): additional options for this function.
                For more details, see the corresponding engine's documentation.

        Returns:
            ArrayLike: The copied array.
        """
        return self.engine.identity(array, **kwargs)

    def cumsum(self, array: ArrayLike, **kwargs) -> ArrayLike:
        """Return the cumulative sum of the elements in ``array``.

        Args:
            array (ArrayLike): input array.
            kwargs (optional): additional options for this function.
                For more details, see the corresponding engine's documentation.

        Returns:
            ArrayLike: New array with the cumulative sum of ``array``.
        """
        return self.engine.math.cumsum(array, **kwargs)

    def default_rng(self, seed: Optional[int] = None):
        """Create a new random number Generator using the engine's default setting.

        Args:
            seed (int, optional): a seed to initialize the generator. If ``None``,
            a random integer is chosen. Defaults to ``None``.

        Returns:
            ArrayLike: The initialized random number generator.
        """
        if seed is None:
            return self.engine.random.Generator.from_non_deterministic_state()

        return self.engine.random.Generator.from_seed(seed)

    def diag(self, array: ArrayLike, **kwargs) -> ArrayLike:
        """Extract or construct a diagonal array.

        If ``array`` is one-dimensional, construct a two-dimensional array with ``array`` as its
        diagonal. Otherwise, extract the diagonal elements of ``array``.

        Args:
            array (ArrayLike): input array.
            kwargs (optional): additional options for this function.
                For more details, see the corresponding engine's documentation.

        Returns:
            ArrayLike: The extracted or constructed diagonal array.
        """
        if len(array.shape) == 1:
            return self.engine.linalg.tensor_diag(array, **kwargs)

        return self.engine.linalg.tensor_diag_part(array, **kwargs)

    def expm(self, array: ArrayLike) -> ArrayLike:
        """Compute the matrix exponential of an ``array``.

        Args:
            array (ArrayLike): input array.

        Returns:
            ArrayLike: The resulting matrix exponential.
        """
        return self.engine.linalg.expm(array)

    def flatnonzero(self, array: ArrayLike) -> ArrayLike:
        """Return indices that are non-zero in the flattened version of ``array``.

        Args:
            array (ArrayLike): input array.

        Returns:
            ArrayLike: Indices of the nonzero elements of flattened ``array``.
        """
        return np.flatnonzero(array)

    def full(
        self,
        shape: Union[int, Tuple[int, ...], List[int]],
        fill_value: Union[complex, float, int],
        **kwargs,
    ) -> ArrayLike:
        """Return a new array with a given ``shape`` filled with ``fill_value``.

        Args:
            shape (int or Tuple[int, ...]): shape of the new array.
            fill_value (complex or float or int): fill value.
            kwargs (optional): additional options for this function.
                For more details, see the corresponding engine's documentation.

        Returns:
            ArrayLike: Array of ``fill_value`` with the given ``shape``.
        """
        return self.engine.fill(shape, fill_value, **kwargs)

    def imag(self, array: ArrayLike) -> Union[int, float, ArrayLike]:
        """Return the element-wise imaginary part of a complex-valued ``array``.

        Args:
            array (ArrayLike): input array.

        Returns:
            int or float or ArrayLike: The element-wise imaginary part of the complex ``array``.
        """
        return self.engine.math.imag(array)

    def isnan(self, array: ArrayLike, **kwargs) -> ArrayLike:
        """Test element-wise for NaN and return result as a boolean array.

        Args:
            array (ArrayLike): input array.
            kwargs (optional): additional options for this function.
                For more details, see the corresponding engine's documentation.

        Returns:
            ArrayLike: ``True`` elements where ``array`` is ``nan``, ``False`` otherwise.
        """
        return self.engine.math.is_nan(array, **kwargs)

    def kron(self, array_1: ArrayLike, array_2: ArrayLike) -> ArrayLike:
        """Kronecker product of two arrays.

        Args:
            array_1 (ArrayLike): first array.
            array_2 (ArrayLike): second array.

        Returns:
            ArrayLike: The resulting array.
        """
        return self.engine.experimental.numpy.kron(array_1, array_2)

    def log(self, array: ArrayLike, **kwargs) -> ArrayLike:
        """Calculate the element-wise natural logarithm of ``array``.

        Args:
            array (ArrayLike): input array.
            kwargs (optional): additional options for this function.
                For more details, see the corresponding engine's documentation.

        Returns:
            float or complex or ArrayLike: Array with the element-wise natural logarithm
            of the elements in `array``.
        """
        return self.engine.math.log(array, **kwargs)

    def log2(self, array: ArrayLike, **kwargs) -> ArrayLike:
        """Calculate the element-wise base-:math:`2` logarithm of ``array``.

        Args:
            array (ArrayLike): input array.
            kwargs (optional): additional options for this function.
                For more details, see the corresponding engine's documentation.

        Returns:
            float or complex or ArrayLike: Array with the element-wise base-:math:`2` logarithm
            of the elements in `array``.
        """
        return self.engine.math.log(array, **kwargs) / self.engine.math.log(2.0)

    def log10(self, array: ArrayLike, **kwargs) -> ArrayLike:
        """Calculate the element-wise base-:math:`10` logarithm of ``array``.

        Args:
            array (ArrayLike): input array.
            kwargs (optional): additional options for this function.
                For more details, see the corresponding engine's documentation.

        Returns:
            float or complex or ArrayLike: Array with the element-wise base-:math:`10` logarithm
            of the elements in `array``.
        """
        return self.engine.math.log(array, **kwargs) / self.engine.math.log(10.0)

    def matrix_norm(
        self, array: ArrayLike, order: Union[int, float, str] = "nuc", **kwargs
    ) -> Union[ArrayLike, float, int]:
        """Calculate norm of a two-dimensional array.

        Args:
            array (ArrayLike): input array.
            order (int or float or str, optional): Order of the norm. For the specific options,
                we refer to the engine's official documentation. Defaults to ``"nuc"``.
            kwargs (optional): additional options for this function.
                For more details, see the corresponding engine's documentation.

        Returns:
            float or ArrayLike: Norm of the matrix.
        """
        array = self.cast(array, dtype=array.dtype)

        if order == "nuc":
            return self.trace(array)

        return self.engine.norm(array, ord=order, **kwargs)

    def max(
        self, array: ArrayLike, **kwargs
    ) -> Union[float, int, ArrayLike]:  # pragma: no cover
        """Return the maximum value of an ``array``.

        Args:
            array (ArrayLike): input array.
            kwargs (optional): additional options for this function.
                For more details, see the corresponding engine's documentation.

        Returns:
            float or int or complex or ArrayLike: Maximum of ``array``, possibly along
            an axis or axes.
        """
        return self.engine.math.reduce_max(array, **kwargs)

    def maximum(self, array_1: ArrayLike, array_2: ArrayLike, **kwargs) -> ArrayLike:
        """Return the element-wise maximum between two arrays.

        Args:
            array_1 (ArrayLike): first array.
            array_2 (ArrayLike): second array.
            kwargs (optional): additional options for this function.
                For more details, see the corresponding engine's documentation.

        Returns:
            ArrayLike: The maximum between ``array_1`` and ``array_2``, element-wise.
        """
        return self.engine.math.maximum(array_1, array_2, **kwargs)

    def min(
        self, array: ArrayLike, **kwargs
    ) -> Union[float, int, ArrayLike]:  # pragma: no cover
        """Return the minimum value of an ``array``.

        Args:
            array (ArrayLike): input array.
            kwargs (optional): additional options for this function.
                For more details, see the corresponding engine's documentation.

        Returns:
            float or int or complex or ArrayLike: Minimum of ``array``, possibly along
            an axis or axes.
        """
        return self.engine.math.reduce_min(array, **kwargs)

    def minimum(self, array_1: ArrayLike, array_2: ArrayLike, **kwargs) -> ArrayLike:
        """Return the element-wise minimum between two arrays.

        Args:
            array_1 (ArrayLike): first array.
            array_2 (ArrayLike): second array.
            kwargs (optional): additional options for this function.
                For more details, see the corresponding engine's documentation.

        Returns:
            ArrayLike: The minimum between ``array_1`` and ``array_2``, element-wise.
        """
        return self.engine.math.minimum(array_1, array_2, **kwargs)

    def mod(self, dividend: ArrayLike, divisor: ArrayLike, **kwargs) -> ArrayLike:
        """Return the element-wise remainder of division.

        Args:
            dividend (ArrayLike): dividend array.
            divisor (float or int or ArrayLike): divisir array.
            kwargs (optional): additional options for this function.
                For more details, see the corresponding engine's documentation.

        Returns:
            ArrayLike: The element-wise remainder of the quotient ``dividend / divisor``.
        """
        return self.engine.math.floormod(dividend, divisor, **kwargs)

    def outer(self, array_1: ArrayLike, array_2: ArrayLike) -> ArrayLike:
        """Calculate the outer product of two arrays.

        Args:
            array_1 (ArrayLike): first array,
            array_2 (ArrayLike): second array.

        Returns:
            ArrayLike: The resulting array.
        """
        return self.tensordot(array_1, array_2, axes=0)

    def prod(self, array: ArrayLike, **kwargs) -> ArrayLike:
        """Return the product of array elements.

        Args:
            array (ArrayLike): input array.
            kwargs (optional): additional options for this function.
                For more details, see the corresponding engine's documentation.

        Returns:
            ArrayLike: New array with the product of elements of ``array``.
        """
        return self.engine.math.reduce_prod(array, **kwargs)

    def random_normal(
        self,
        mean: Union[float, int],
        stddev: Union[float, int],
        size: Optional[Union[int, List[int], Tuple[int, ...]]] = None,
        seed: Optional[int] = None,
        dtype: Optional[DTypeLike] = None,
    ) -> ArrayLike:
        """Generate random numbers from a normal (Gaussian) distribution.

        Args:
            mean (float or int): mean value of the Gaussian distribution.
            stddev (float or int): standard deviation of the Gaussian distribution.
            size (int or List[int] or Tuple[int, ...], optional): output shape. If ``None``,
                a single sample is returned. Defaults to ``None``.
            seed (int, optional): a seed to initialize the random number generator. If ``None``,
                a random integer is chosen. Defaults to ``None``.
            dtype (DTypeLike, optional): data type of the resulting array. If ``None``,
                defaults to the global data type of the ``Backend``. Defaults to ``None``.

        Returns:
            ArrayLike: The generated random sample(s).
        """
        if isinstance(size, int):
            size = [size]

        if dtype is None:
            dtype = self.float64

        if seed is not None:  # pragma: no cover
            local_state = self.default_rng(seed) if isinstance(seed, int) else seed

            # local rng usually only has standard normal implemented
            distribution = local_state.standard_normal(size)
            distribution *= stddev
            distribution += mean

            return self.cast(distribution, dtype=dtype)

        return self.engine.random.normal(size, mean, stddev, dtype=dtype)

    def random_uniform(
        self,
        low: Union[float, int] = 0.0,
        high: Union[float, int] = 1.0,
        size: Optional[Union[int, Tuple[int, ...]]] = None,
        seed: Optional[int] = None,
        **kwargs,
    ) -> ArrayLike:
        """Generate random numbers in the interval ``[low, high)`` over the uniform distribution.

        Args:
            low (float or int, optional):  lower integer in the interval (inclusive).
                Defaults to :math:`0.0`.
            high (float or int, optional): highest integer in the interval (exclusive).
                Defaults to :math:`1.0`.
            size (int or Tuple[int, ...], optional): output shape. If ``None``,
                a single sample is returned. Defaults to ``None``.
            seed (int, optional): a seed to initialize the random number generator. If ``None``,
                a random integer is chosen. Defaults to ``None``.
            kwargs (optional): additional options for this function.
                For more details, see the corresponding engine's documentation.

        Returns:
            ArrayLike: The generated random sample(s).
        """
        dtype = kwargs.get("dtype", self.float64)

        shape = (size,) if isinstance(size, int) else size

        if seed is not None:
            local_state = self.default_rng(seed) if isinstance(seed, int) else seed

            return self.cast(
                local_state.uniform(shape=shape, minval=low, maxval=high), dtype=dtype
            )

        return self.cast(
            self.engine.random.uniform(shape=shape, minval=low, maxval=high),
            dtype=dtype,
        )

    def ravel(self, array: ArrayLike, **kwargs) -> ArrayLike:  # pragma: no cover
        """Return a contiguous flattened ``array``.

        Args:
            array (ArrayLike): input array.
            kwargs (optional): additional options for this function.
                For more details, see the corresponding engine's documentation.

        Returns:
            ArrayLike: Contiguous, one-dimensional array with the flattened elements of ``array``.
        """
        return self.engine.keras.ops.ravel(array, **kwargs)  # pylint: disable=E1101

    def real(self, array: ArrayLike) -> ArrayLike:
        """Return the element-wise real part of a complex-valued ``array``.

        Args:
            array (ArrayLike): input array.

        Returns:
            int or float or ArrayLike: The element-wise real part of the complex ``array``.
        """

        return self.engine.math.real(array)

    def round(self, array: ArrayLike, decimals: int = 0) -> ArrayLike:
        """Return element-wise evenly round ``array`` to the given number of ``decimals``.

        Args:
            array (ArrayLike): input array.
            decimals (int, optional): number of decimal places to round to. Defaults to :math:`0`.
            kwargs (optional): additional options for this function.
                For more details, see the corresponding engine's documentation.

        Returns:
            ArrayLike: The rounded array.
        """
        return self.engine.experimental.numpy.around(array, decimals=decimals)

    def sin(self, array: ArrayLike, **kwargs) -> ArrayLike:
        """Calculate the element-wise sine of values in ``array``.

        Args:
            array (ArrayLike): array to calculate ``sine`` of.
            kwargs (optional): additional options for this function.
                For more details, see the corresponding engine's documentation.

        Returns:
            ArrayLike: Array with the element-wise sine of elements in `array``.
        """
        return self.engine.math.sin(array, **kwargs)

    def sqrt(self, array: ArrayLike) -> ArrayLike:
        """Return the element-wise non-negative square-root of ``array``.

        Args:
            array (ArrayLike): input array.

        Returns:
            ArrayLike: Array containing the positive square-root of each element in ``array``.
        """
        return self.engine.math.sqrt(array)

    def sum(
        self, array: ArrayLike, axis: Optional[Tuple[int, ...]] = None, **kwargs
    ) -> ArrayLike:
        """Sum of ``array`` elements over a given ``axis``.

        Args:
            array (ArrayLike): input array.
            axis (int, optional): axis or axes along which a sum is performed. If ``None``,
                will sum all of the elements of the input ``array``. Defaults to ``None``.
            kwargs (optional): additional options for this function.
                For more details, see the corresponding engine's documentation.

        Returns:
            int or float or complex or ArrayLike: An array with the resulting sum
            of elements of ``array``.
        """
        return self.engine.math.reduce_sum(array, axis, **kwargs)

    def trace(self, array: ArrayLike) -> Union[int, float]:
        """Return the sum along diagonals of the array.

        Args:
            array (ArrayLike): input array.

        Returns:
            int or float: The sum along the diagonal.
        """
        return self.engine.linalg.trace(array)

    def var(
        self, array: ArrayLike, **kwargs
    ) -> Union[float, int, ArrayLike]:  # pragma: no cover
        """Calculate the variance of ``array``.

        Args:
            array (ArrayLike): input array.
            kwargs (optional): additional options for this function.
                For more details, see the corresponding engine's documentation.

        Returns:
            float or ArrayLike: A new array containing the variance.
        """
        return self.engine.math.reduce_variance(array, **kwargs)

    def vector_norm(
        self,
        state: ArrayLike,
        order: Union[int, float, str] = 2,
        dtype: Optional[DTypeLike] = None,
        **kwargs,
    ) -> Union[ArrayLike, float, int]:
        """Calculate norm of an one-dimensional ``array``.

        Args:
            array (ArrayLike): input array.
            order (int or float or str, optional): Order of the norm. For the specific options,
                we refer to the engine's official documentation. Defaults to :math:`2`.
            dtype (DTypeLike, optional): data type of the resulting array. If ``None``,
                defaults to the global data type of the ``Backend``. Defaults to ``None``.
            kwargs (optional): additional options for this function.
                For more details, see the corresponding engine's documentation.

        Returns:
            float or ArrayLike: Norm of the ``array``.
        """
        if dtype is None:
            dtype = self.dtype

        state = self.cast(state, dtype=dtype)

        return self.real(self.engine.norm(state, ord=order, **kwargs))

    def vstack(self, arrays: ArrayLike, **kwargs) -> ArrayLike:
        """Stack ``arrays`` in sequence vertically (row wise).

        Args:
            arrays (Tuple[ArrayLike, ...]): arrays to be stacked.
            kwargs (optional): additional options for this function.
                For more details, see the corresponding engine's documentation.

        Returns:
            ArrayLike: The array formed by stacking the given ``arrays``.
        """
        return self.engine.stack(arrays, axis=0, **kwargs)

    ########################################################################################
    ######## Methods related to linear algebra operations                           ########
    ########################################################################################

    def matrix_exp(
        self,
        matrix: ArrayLike,
        phase: Union[float, int, complex] = 1,
        eigenvectors: Optional[ArrayLike] = None,
        eigenvalues: Optional[ArrayLike] = None,
    ) -> ArrayLike:
        if eigenvectors is None or self.is_sparse(matrix):
            return self.expm(phase * matrix)
        return super().matrix_exp(matrix, phase, eigenvectors, eigenvalues)

    def matrix_power(
        self,
        matrix: ArrayLike,
        power: Union[float, int],
        precision_singularity: float = 1e-14,
        dtype: Optional[DTypeLike] = None,
    ) -> ArrayLike:
        if not isinstance(power, (float, int)):
            raise_error(
                TypeError,
                f"``power`` must be either float or int, but it is type {type(power)}.",
            )

        if dtype is None:
            dtype = self.dtype

        if power < 0.0:
            # negative powers of singular matrices via SVD
            determinant = self.det(matrix)
            if abs(determinant) < precision_singularity:
                return self._negative_power_singular_matrix(
                    matrix, power, precision_singularity, dtype
                )

        return super().matrix_power(matrix, power, precision_singularity)

    def singular_value_decomposition(self, array: ArrayLike) -> ArrayLike:
        # needed to unify order of return
        s_matrix, u_matrix, v_matrix = self.engine.linalg.svd(array)
        return u_matrix, s_matrix, self.conj(self.transpose(v_matrix))

    def jacobian(
        self,
        circuit: Circuit,
        parameters: Optional[ArrayLike] = None,
        initial_state: Optional[ArrayLike] = None,
        return_complex: bool = True,
    ) -> ArrayLike:
        copied = circuit.copy(deep=True)

        # necessary for the tape to properly watch the variables
        parameters = self.engine.Variable(parameters)

        with self.engine.GradientTape(persistent=return_complex) as tape:
            copied.set_parameters(parameters)
            state = self.execute_circuit(copied, initial_state=initial_state).state()
            real = self.real(state)
            if return_complex:
                imag = self.imag(state)

        if return_complex:
            return tape.jacobian(real, parameters), tape.jacobian(imag, parameters)

        return tape.jacobian(real, parameters)

    def zero_state(
        self,
        nqubits: int,
        density_matrix: bool = False,
        dtype: Optional[DTypeLike] = None,
    ) -> ArrayLike:
        if dtype is None:
            dtype = self.dtype

        shape = [[0, 0]] if density_matrix else [[0]]
        idx = self.engine.constant(shape, dtype="int32")

        shape = 2 * (2**nqubits,) if density_matrix else (2**nqubits,)
        state = self.zeros(shape, dtype=dtype)

        update = self.engine.constant([1], dtype=dtype)

        state = self.engine.tensor_scatter_nd_update(state, idx, update)

        return state

    ########################################################################################
    ######## Methods related to circuit execution                                   ########
    ########################################################################################

    def execute_circuit(
        self,
        circuit: Circuit,
        initial_state: Optional[ArrayLike] = None,
        nshots: int = 1000,
    ) -> Union[CircuitResult, MeasurementOutcomes, QuantumState]:
        with self.engine.device(self.device):
            return super().execute_circuit(circuit, initial_state, nshots)

    def execute_circuit_repeated(
        self, circuit: Circuit, nshots: int, initial_state: Optional[ArrayLike] = None
    ) -> Union[CircuitResult, MeasurementOutcomes, QuantumState]:
        with self.engine.device(self.device):
            return super().execute_circuit_repeated(circuit, nshots, initial_state)

    def matrix(self, gate: Gate) -> ArrayLike:
        npmatrix = super().matrix(gate)
        # delete cached matrix if it's symbolic
        if self.engine.is_symbolic_tensor(npmatrix):
            delattr(self.matrices, gate.__class__.__name__)
        return npmatrix

    def matrix_fused(self, fgate: Gate) -> ArrayLike:
        rank = len(fgate.target_qubits)
        # tf only supports coo sparse arrays
        # however they are probably not as efficient as csr ones
        # at this point it is maybe better to just use dense arrays
        # from the tests no major performance difference emerged
        # with dense tensors, thus keeping sparse representation for now
        matrix = self.identity(2**rank, dtype=self.dtype)

        for gate in fgate.gates:
            gmatrix = gate.matrix(self)
            # add controls if controls were instantiated using
            # the ``Gate.controlled_by`` method
            num_controls = len(gate.control_qubits)
            if num_controls > 0:
                gmatrix = self.engine.linalg.LinearOperatorBlockDiag(
                    self.identity(2 ** len(gate.qubits) - len(gmatrix)), gmatrix
                )
            # Kronecker product with identity is needed to make the
            # original matrix have shape (2**rank x 2**rank)
            eye = self.identity(2 ** (rank - len(gate.qubits)))
            gmatrix = self.kron(gmatrix, eye)
            # Transpose the new matrix indices so that it targets the
            # target qubits of the original gate
            original_shape = gmatrix.shape
            gmatrix = self.reshape(gmatrix, 2 * rank * (2,))
            qubits = list(gate.qubits)
            indices = qubits + [q for q in fgate.target_qubits if q not in qubits]
            indices = np.argsort(indices)
            transpose_indices = list(indices)
            transpose_indices.extend(indices + rank)
            gmatrix = self.transpose(gmatrix, transpose_indices)
            gmatrix = self.reshape(gmatrix, original_shape)
            # fuse the individual gate matrix to the total ``FusedGate`` matrix
            # we are using sparse matrices to improve perfomances
            if "cpu" in self.device.lower():
                matrix = self.engine.sparse.sparse_dense_matmul(
                    self.engine.sparse.from_dense(gmatrix), matrix
                )
            else:  # pragma: no cover
                matrix = self.matmul(gmatrix, matrix)

        return matrix

    def matrix_parametrized(self, gate: Gate) -> ArrayLike:
        npmatrix = super().matrix_parametrized(gate)
        return self.cast(npmatrix, dtype=self.dtype)

    ########################################################################################
    ######## Methods related to the execution and post-processing of measurements   ########
    ########################################################################################

    def calculate_frequencies(self, samples: ArrayLike) -> Counter:
        # redefining this because ``tnp.unique`` is not available
        res, _, counts = self.engine.unique_with_counts(samples, out_idx="int64")
        res, counts = np.array(res), np.array(counts)
        if res.dtype == "string":
            res = [r.numpy().decode("utf8") for r in res]
        else:
            res = [int(r) for r in res]
        return Counter({k: int(v) for k, v in zip(res, counts)})

    def sample_shots(self, probabilities: ArrayLike, nshots: int) -> ArrayLike:
        # redefining this because ``tnp.random.choice`` is not available
        logits = self.log(probabilities)[self.engine.newaxis]
        samples = self.engine.random.categorical(logits, nshots)[0]
        return samples

    def samples_to_binary(self, samples: ArrayLike, nqubits: int) -> ArrayLike:
        # redefining this because ``tnp.right_shift`` is not available
        qrange = self.engine.range(nqubits - 1, -1, -1, dtype=self.int32)
        samples = self.cast(samples, dtype=self.int32)
        samples = self.engine.bitwise.right_shift(samples[:, np.newaxis], qrange)
        return samples % 2

    def update_frequencies(
        self, frequencies: ArrayLike, probabilities: ArrayLike, nsamples: int
    ) -> ArrayLike:
        # redefining this because ``tnp.unique`` and tensor update is not available
        samples = self.sample_shots(probabilities, nsamples)
        res, _, counts = self.engine.unique_with_counts(samples, out_idx=self.int64)
        frequencies = self.engine.tensor_scatter_nd_add(
            frequencies, res[:, self.engine.newaxis], counts
        )
        return frequencies

    def assert_allclose(
        self, value: ArrayLike, target: ArrayLike, rtol: float = 1e-7, atol: float = 0.0
    ) -> None:  # pragma: no cover
        if isinstance(value, (CircuitResult, QuantumState)):
            value = value.state()
        if isinstance(target, (CircuitResult, QuantumState)):
            target = target.state()

        np.testing.assert_allclose(value, target, rtol=rtol, atol=atol)

    def _test_regressions(self, name):
        if name == "test_measurementresult_apply_bitflips":
            return [
                [4, 0, 0, 1, 0, 0, 1, 0, 0, 0],
                [0, 1, 1, 2, 1, 1, 4, 0, 0, 4],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 4, 0, 0, 0, 4],
            ]

        if name == "test_probabilistic_measurement":
            if "GPU" in self.device:  # pragma: no cover
                return {0: 273, 1: 233, 2: 242, 3: 252}

            return {0: 271, 1: 239, 2: 242, 3: 248}

        if name == "test_unbalanced_probabilistic_measurement":
            if "GPU" in self.device:  # pragma: no cover
                return {0: 196, 1: 153, 2: 156, 3: 495}

            return {0: 168, 1: 188, 2: 154, 3: 490}

        return [
            {5: 30},
            {5: 12, 7: 6, 4: 6, 1: 5, 6: 1},
            {3: 7, 6: 4, 2: 4, 7: 4, 0: 4, 5: 3, 4: 2, 1: 2},
        ]
