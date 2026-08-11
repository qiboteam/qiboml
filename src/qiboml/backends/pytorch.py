"""Module defining the PyTorch backend."""

from typing import List, Optional, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike, DTypeLike
from qibo import Circuit, __version__
from qibo.backends.abstract import Backend
from qibo.backends.npmatrices import NumpyMatrices
from qibo.config import raise_error
from qibo.gates.abstract import Gate
from qibo.result import CircuitResult, QuantumState


class TorchMatrices(NumpyMatrices):
    """Matrix representation of every gate as a torch Tensor.

    Args:
        dtype (torch.dtype): Data type of the matrices.
    """

    def __init__(self, dtype: DTypeLike, device: str):
        import torch  # pylint: disable=import-outside-toplevel  # type: ignore

        super().__init__(dtype)
        self.engine = torch
        self.dtype = dtype
        self.device = device

    def _cast(
        self, array: ArrayLike, dtype: DTypeLike, device: Optional[str] = None
    ) -> ArrayLike:
        if device is None:
            device = self.device

        flattened = [item for sublist in array for item in sublist]
        tensor_list = [
            self.engine.as_tensor(elem, dtype=dtype, device=device)
            for elem in flattened
        ]

        return self.engine.stack(tensor_list).reshape(len(array), len(array))

    def I(self, n: int = 2) -> ArrayLike:
        return self.engine.eye(n, dtype=self.dtype, device=self.device)

    def Unitary(self, u: ArrayLike) -> ArrayLike:
        return self._cast(u, dtype=self.dtype, device=self.device)


class PyTorchBackend(Backend):
    def __init__(self):
        super().__init__()
        import torch  # pylint: disable=import-outside-toplevel  # type: ignore

        self.engine = torch

        self.name = "qiboml"
        self.platform = "pytorch"
        self.versions = {
            "qibo": __version__,
            "numpy": np.__version__,
            "torch": self.engine.__version__,
        }

        self.device = self.engine.get_default_device()
        self.dtype = self.complex128
        self.matrices = TorchMatrices(self.dtype, self.device)
        self.nthreads = 1
        self.numeric_types += (
            self.int8,
            self.int32,
            self.int64,
            self.float32,
            self.float64,
            self.complex64,
            self.complex128,
        )
        self.parameter_dtype = self.float64
        self.tensor_types = (self.engine.Tensor, np.ndarray)

    def cast(
        self,
        array: ArrayLike,
        dtype: Optional[DTypeLike] = None,
        copy: bool = False,
        device: Optional[str] = None,
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
            device (str, optional): used to switch ``torch.Tensor`` to and from CPUs and GPUs.
                Please see the official ``pytorch`` documentation. If ``None``, uses the default
                ``Backend`` device. Defaults to ``None``.

        Returns:
            ArrayLike: ``array`` casted to ``dtype`` and ``device``, possibly copied in memory.
        """

        if dtype is None:
            dtype = self.dtype

        if device is None:
            device = self.device

        if isinstance(array, self.engine.Tensor):
            array = array.to(dtype)
        elif (
            isinstance(array, list)
            and len(array) > 0
            and all(isinstance(row, self.engine.Tensor) for row in array)
        ):
            array = self.engine.stack(array)
        else:
            array = self.engine.tensor(array, dtype=dtype)

        if copy:
            return array.clone().to(device)

        return array.to(device)

    def is_sparse(self, array: ArrayLike) -> bool:
        """Determine if a given array is a sparse tensor.

        Args:
            array (ArrayLike): array to determine the sparsity of.

        Returns:
            bool: ``True`` if ``array`` is sparse, ``False`` otherwise.
        """
        if isinstance(array, self.engine.Tensor):
            return array.is_sparse

        return super().is_sparse(array)

    def set_device(self, device: str) -> None:  # pragma: no cover
        """Set simulation device. Works in-place.

        Args:
            device (str): Device index, *e.g.* ``/CPU:0`` for CPU, or ``/GPU:1`` for
                the second GPU in a multi-GPU environment.
        """
        if "CPU" in device:
            self.device = "cpu"
        else:
            self.device = "cuda:" + device.split(":")[-1]

    def set_dtype(self, dtype: str) -> None:
        """Set data type of arrays created using the backend. Works in-place.

        .. note::
            The data types ``float32`` and ``float64`` are intended to be used when the circuits
            to be simulated only contain gates with real-valued matrix representations.
            Using one of the aforementioned data types with circuits that contain complex-valued
            matrices will raise a casting error.

        .. note::
            List of gates that always admit a real-valued matrix representation:
            :class:`qibo.gates.I`, :class:`qibo.gates.X`, :class:`qibo.gates.Z`,
            :class:`qibo.gates.H`, :class:`qibo.gates.Align`, :class:`qibo.gates.RY`,
            :class:`qibo.gates.CNOT`, :class:`qibo.gates.CZ`, :class:`qibo.gates.CRY`,
            :class:`qibo.gates.SWAP`, :class:`qibo.gates.FSWAP`, :class:`qibo.gates.GIVENS`,
            :class:`qibo.gates.RBS`, :class:`qibo.gates.TOFFOLI`, :class:`qibo.gates.CCZ`,
            and :class:`qibo.gates.FanOut`.

        .. note::
            The following parametrized gates can have real-valued matrix representations
            depending on the values of their parameters:
            :class:`qibo.gates.RX`, :class:`qibo.gates.RZ`, :class:`qibo.gates.U1`,
            :class:`qibo.gates.U2`, :class:`qibo.gates.U3`, :class:`qibo.gates.CRX`,
            :class:`qibo.gates.CRZ`, :class:`qibo.gates.CU1`, :class:`qibo.gates.CU2`,
            :class:`qibo.gates.CU3`, :class:`qibo.gates.fSim`, :class:`qibo.gates.GeneralizedfSim`,
            :class:`qibo.gates.RXX`, :class:`qibo.gates.RYY`, :class:`qibo.gates.RZZ`,
            :class:`qibo.gates.RZX`, and :class:`qibo.gates.GeneralizedRBS`.

        Args:
            dtype (str): the options are the following: ``complex128``, ``complex64``,
                ``float64``, and ``float32``.
        """
        dtypes_str = ("float32", "float64", "complex64", "complex128")

        if dtype not in self.numeric_types and dtype not in dtypes_str:
            raise_error(
                ValueError,
                f"Unknown ``dtype`` ``{dtype}``. For this backend ({self}), "
                + f"``dtype`` must be either one of the following string: {dtypes_str}, "
                + f"or one of the following options: {self.numeric_types}",
            )

        if dtype != self.dtype:
            self.dtype = dtype

            if self.matrices is not None:
                self.matrices = self.matrices.__class__(
                    getattr(self, self.dtype), device=self.device
                )

    def set_seed(self, seed: int) -> None:
        """Set the seed of the random number generator. Works in-place.

        Args:
            seed (int or None): seed to be set. If ``None``, seed is random.
        """
        if seed is None:
            seed = self.engine.seed()
        self.engine.manual_seed(seed)

    def set_threads(self, nthreads: int) -> None:  # pragma: no cover
        """Set number of threads for CPU backend simulations that accept it. Works in-place.

        Args:
            nthreads (int): Number of threads.
        """
        self.engine.set_num_threads(nthreads)

    def to_numpy(self, array: ArrayLike) -> ArrayLike:
        """Convert ``array`` to a ``numpy.ndarray``.

        Args:
            array (ArrayLike): array to be converted to ``numpy.ndarray``.

        Returns:
            ArrayLike: Original array converted to ``numpy.ndarray``.
        """
        if isinstance(array, list):
            return np.asarray([self.to_numpy(i) for i in array])

        if isinstance(array, self.engine.Tensor):
            if array.requires_grad:
                return array.detach().cpu().numpy()

            return array.cpu().numpy(force=True)

        return array

    ########################################################################################
    ######## Methods related to array manipulation                                  ########
    ########################################################################################

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
        array = self.cast(array, dtype=array.dtype)
        return array.to_sparse_coo(**kwargs)

    def copy(self, array: ArrayLike, **kwargs) -> ArrayLike:
        """Return a copy of ``array``.

        Args:
            array (ArrayLike): input array.
            kwargs (optional): additional options for this function.
                For more details, see the corresponding engine's documentation.

        Returns:
            ArrayLike: The copied array.
        """
        return self.engine.clone(array, **kwargs)

    def csr_matrix(self, array: ArrayLike, **kwargs):  # pragma: no cover
        """Return the sparse version of ``array`` in compressed sparse row format.

        Args:
            array (ArrayLike): input array.
            kwargs (optional): additional options for this function.
                For more details, see the corresponding engine's documentation.

        Returns:
            ArrayLike: The compressed-sparse-row version of ``array``.
        """
        array = self.cast(array, dtype=array.dtype)
        return array.to_sparse_csr(**kwargs)

    def default_rng(self, seed: Optional[int] = None):
        """Create a new random number Generator using the engine's default setting.

        Args:
            seed (int, optional): a seed to initialize the generator. If ``None``,
            a random integer is chosen. Defaults to ``None``.

        Returns:
            ArrayLike: The initialized random number generator.
        """
        if seed is not None:
            if isinstance(seed, int):
                default_rng = self.engine.Generator()
                default_rng.manual_seed(seed)
            else:
                default_rng = seed
        else:
            default_rng = self.engine.Generator()

        return default_rng

    def expand_dims(
        self, array: ArrayLike, axis: Union[int, Tuple[int, ...]]
    ) -> ArrayLike:
        """Expand the shape of an ``array`` along an ``axis``.

        Insert a new ``axis`` that will appear at the ``axis`` position in the expanded
        ``array`` shape.

        Args:
            array (ArrayLike): input array.
            axis (int or Tuple[int, ...]): Position in the expanded axes where the new axis
                (or axes) is placed.

        Returns:
            ArrayLike: Copy of ``array`` with expanded dimensions along ``axis``.
        """
        return self.engine.unsqueeze(array, axis)

    def expm(self, array: ArrayLike) -> ArrayLike:
        """Compute the matrix exponential of an ``array``.

        Args:
            array (ArrayLike): input array.

        Returns:
            ArrayLike: The resulting matrix exponential.
        """
        return self.engine.linalg.matrix_exp(array)  # pylint: disable=not-callable

    def flatnonzero(self, array: ArrayLike) -> ArrayLike:
        """Return indices that are non-zero in the flattened version of ``array``.

        Args:
            array (ArrayLike): input array.

        Returns:
            ArrayLike: Indices of the nonzero elements of flattened ``array``.
        """
        return self.engine.nonzero(array).flatten()

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
        return self.engine.fmod(dividend, divisor, **kwargs)

    def nonzero(self, array: ArrayLike) -> ArrayLike:
        """Return the indices of the elements of ``array`` that are non-zero.

        Args:
            array (ArrayLike): input array.

        Returns:
            ArrayLike: Array with the indices of the non-zero elements of ``array``.
        """
        return self.engine.nonzero(array, as_tuple=True)

    def random_choice(
        self,
        array: ArrayLike,
        size: Optional[Union[int, Tuple[int, ...]]] = None,
        replace: bool = True,
        p: Optional[ArrayLike] = None,
        seed: Optional[int] = None,
        **kwargs,
    ) -> ArrayLike:
        """Generate random sample(s) from a given one-dimensional ``array``
        over the probability distribution ``p``.

        Args:
            array (ArrayLike): If a tensor, a random sample is generated from its elements.
                If an ``int``, the random sample is generated as if it were
                ``Backend.arange(array)``.
            size (int or Tuple[int, ...], optional): output shape. If ``None``,
                a single sample is returned. Defaults to ``None``.
            replace (bool, optional): If ``True``, the values in ``array`` can be
                sampled multiple times. If ``False``, values are only sampled once.
                Defaults to ``True``.
            p (ArrayLike, optional): probabilities associated with each entry in ``array``.
                If ``None``, defaults to the uniform distribution. Defaults to ``None``.
            seed (int, optional): a seed to initialize the random number generator. If ``None``,
                a random integer is chosen. Defaults to ``None``.
            kwargs (optional): additional options for this function.
                For more details, see the corresponding engine's documentation.

        Returns:
            ArrayLike: The generated random sample(s).
        """
        dtype = kwargs.get("dtype", self.float64)

        if size is None:
            size = 1

        if seed is not None:
            local_state = self.default_rng(seed) if isinstance(seed, int) else seed

            indices = self.engine.multinomial(
                p, num_samples=size, replacement=replace, generator=local_state
            )

            return self.cast(array[indices], dtype=dtype, copy=True)

        indices = self.engine.multinomial(p, num_samples=size, replacement=replace)

        return self.cast(array[list(indices)], dtype=dtype, copy=True)

    def random_integers(
        self,
        low: int,
        high: Optional[int] = None,
        size: Optional[Union[int, Tuple[int, ...]]] = None,
        seed: Optional[int] = None,
        **kwargs,
    ) -> ArrayLike:
        """Generate random integers in the interval ``[low, high)`` over the uniform distribution.

        Args:
            low (int): lower integer in the interval (inclusive). If ``high`` is ``None``,
                then ``high`` :math:`\\leftarrow` ``low``, and ``low`` defaults to :math:`0`.
            high (Optional[int], optional): if not ``None``, then highest integer in the interval
                (exclusive). If ``None``, then ``high`` :math:`\\leftarrow` ``low``, and ``low``
                defaults to :math:`0`. Defaults to ``None``.
            size (int or Tuple[int, ...], optional): output shape. If ``None``,
                a single sample is returned. Defaults to ``None``.
            seed (int, optional): a seed to initialize the random number generator. If ``None``,
                a random integer is chosen. Defaults to ``None``.
            kwargs (optional): additional options for this function.
                For more details, see the corresponding engine's documentation.

        Returns:
            ArrayLike: The generated random sample(s).
        """
        dtype = kwargs.get("dtype", self.int64)

        if high is None:
            high = low
            low = 0

        if size is None:
            size = (1,)
        elif isinstance(size, int):
            size = (size,)

        if seed is not None:
            local_state = self.default_rng(seed) if isinstance(seed, int) else seed

            return self.cast(
                self.engine.randint(low, high, size, generator=local_state), dtype=dtype
            )

        return self.cast(self.engine.randint(low, high, size), dtype=dtype)

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
            size = (size,)

        if dtype is None:
            dtype = self.float64

        if seed is not None:  # pragma: no cover
            local_state = self.default_rng(seed) if isinstance(seed, int) else seed

            # local rng usually only has standard normal implemented
            distribution = local_state.standard_normal(size)
            distribution *= stddev
            distribution += mean

            return self.cast(distribution, dtype=dtype)

        return self.cast(self.engine.normal(mean, stddev, size), dtype=dtype)

    def random_sample(
        self, size: Union[int, Tuple[int, ...]], seed=None, **kwargs
    ) -> ArrayLike:
        """Generate random numbers in the interval :math:`[0.0, \\, 1.0)``
        over the uniform distribution.

        Args:
            size (int or List[int] or Tuple[int, ...], optional): output shape.
            seed (int, optional): a seed to initialize the random number generator. If ``None``,
                a random integer is chosen. Defaults to ``None``.
            kwargs (optional): additional options for this function.
                For more details, see the corresponding engine's documentation.

        Returns:
            ArrayLike: The generated random sample(s).
        """
        dtype = kwargs.get("dtype", self.float64)

        if seed is not None:
            local_state = self.default_rng(seed) if isinstance(seed, int) else seed
        else:
            local_state = None

        return self.cast(self.engine.rand(size, generator=local_state), dtype=dtype)

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

        if seed is not None:
            local_state = self.default_rng(seed) if isinstance(seed, int) else seed

            return self.cast(
                low + (high - low) * self.engine.rand(size, generator=local_state),
                dtype=dtype,
            )

        return self.cast(low + (high - low) * self.engine.rand(size), dtype=dtype)

    def right_shift(self, *args, **kwargs) -> ArrayLike:
        """Shift the bits of an integer to the right

        Args:
            args (int): positional arguments for this function.
                For more details, see the corresponding engine's documentation.
            kwargs (optional): additional options for this function.
                For more details, see the corresponding engine's documentation.

        Returns:
            ArrayLike: The right-shifted array.
        """
        return self.engine.bitwise_right_shift(*args, **kwargs)

    def round(self, array: ArrayLike, decimals: int = 0, **kwargs) -> ArrayLike:
        """Return element-wise evenly round ``array`` to the given number of ``decimals``.

        Args:
            array (ArrayLike): input array.
            decimals (int, optional): number of decimal places to round to. Defaults to :math:`0`.
            kwargs (optional): additional options for this function.
                For more details, see the corresponding engine's documentation.

        Returns:
            ArrayLike: The rounded array.
        """
        len_size = len(array.size())
        dtype = array.dtype if len_size == 0 else array[0].dtype
        if "complex" in str(dtype):
            return self.engine.round(
                self.real(array), decimals=decimals, **kwargs
            ) + 1j * self.engine.round(self.imag(array), decimals=decimals, **kwargs)

        return super().round(array, decimals, **kwargs)

    def transpose(
        self, array: ArrayLike, axes: Union[Tuple[int, ...], List[int]] = None
    ) -> ArrayLike:
        """Return an ``array`` with ``axes`` transposed.

        Args:
            array (ArrayLike): input array.
            axes (Tuple[int, ...] or List[int], optional): axes to be transposed.
                Defaults to ``None``.

        Returns:
            ArrayLike: The resulting transposed array.
        """
        return self.engine.permute(array, axes)

    def tril_indices(
        self, row: int, offset: int = 0, col: Optional[int] = None, **kwargs
    ) -> ArrayLike:
        """Return the indices for the lower-triangle of an ``(row, col)``-dimensional ``array``.

        Args:
            row (int): the row dimension of the arrays for which the returned indices will be valid.
            offset (int, optional): diagonal offset (see :meth:`qibo.backends.Backend.tril`
                for details). Defaults to :math:`0`.
            col (int, optional): the column dimension of the arrays for which the returned arrays
                will be valid. If ``None``, defaults to the same value as ``row``.
                Defaults to ``None``.

        Returns:
            Tuple[ArrayLike, ArrayLike]: The row and column indices, respectively.
        """
        if col is None:
            col = row
        return self.engine.tril_indices(row, col, offset, **kwargs)

    ########################################################################################
    ######## Methods related to linear algebra operations                           ########
    ########################################################################################

    def eigenvalues(
        self, matrix: ArrayLike, k: int = 6, hermitian: bool = True
    ) -> ArrayLike:
        if hermitian:
            return self.eigvalsh(matrix)  # pylint: disable=not-callable
        return self.eigvals(matrix)  # pylint: disable=not-callable

    def eigenvectors(
        self, matrix: ArrayLike, k: int = 6, hermitian: int = True
    ) -> ArrayLike:
        if hermitian:
            return self.eigh(matrix)  # pylint: disable=not-callable
        return self.eig(matrix)  # pylint: disable=not-callable

    def jacobian(
        self,
        circuit: Circuit,
        parameters: Optional[ArrayLike] = None,
        initial_state: Optional[ArrayLike] = None,
        return_complex: bool = True,
    ) -> ArrayLike:
        copied = circuit.copy(deep=True)

        def func(parameters):
            """torch requires object(s) to be wrapped in a function."""
            copied.set_parameters(parameters)
            state = self.execute_circuit(copied, initial_state=initial_state).state()
            if return_complex:
                return self.real(state), self.imag(state)
            return self.real(state)

        return self.engine.autograd.functional.jacobian(func, parameters)

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

        copied = self.cast(matrix, copy=True)  # pylint: disable=E1111
        copied = self.to_numpy(copied) if power >= 0.0 else copied.detach()
        copied = super().matrix_power(  # pylint: disable=E1111
            copied, power, precision_singularity, dtype
        )
        return self.cast(copied, dtype=copied.dtype)  # pylint: disable=E1111

    ########################################################################################
    ######## Methods related to circuit execution                                   ########
    ########################################################################################

    def matrix_fused(self, fgate: Gate) -> ArrayLike:
        rank = len(fgate.target_qubits)
        matrix = self.identity(2**rank, dtype=self.dtype)
        if self.engine.backends.mkl.is_available():
            matrix = matrix.to_sparse_csr()

        for gate in fgate.gates:
            gmatrix = gate.matrix(self)
            # add controls if controls were instantiated using
            # the ``Gate.controlled_by`` method
            num_controls = len(gate.control_qubits)
            if num_controls > 0:
                gmatrix = self.block_diag(  # pylint: disable=E1111
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
            if self.engine.backends.mkl.is_available():
                gmatrix = gmatrix.to_sparse_csr()
            matrix = gmatrix @ matrix

        if self.engine.backends.mkl.is_available():
            return matrix.to_dense()
        return matrix

    def matrix_parametrized(self, gate: Gate) -> ArrayLike:
        """Convert a parametrized gate to its matrix representation in the computational basis."""
        name = gate.__class__.__name__
        _matrix = getattr(self.matrices, name)
        if name == "GeneralizedRBS":
            for parameter in ["theta", "phi"]:
                if not isinstance(gate.init_kwargs[parameter], self.engine.Tensor):
                    gate.init_kwargs[parameter] = self._cast_parameter(
                        gate.init_kwargs[parameter], trainable=gate.trainable
                    )

            _matrix = _matrix(
                qubits_in=gate.init_args[0],
                qubits_out=gate.init_args[1],
                theta=gate.init_kwargs["theta"],
                phi=gate.init_kwargs["phi"],
            )
            return _matrix

        new_parameters = []
        for parameter in gate.parameters:
            if not isinstance(parameter, self.engine.Tensor):
                parameter = self._cast_parameter(parameter, trainable=gate.trainable)
            elif parameter.requires_grad:
                gate.trainable = True
            new_parameters.append(parameter)
        gate.parameters = tuple(new_parameters)
        _matrix = _matrix(*gate.parameters)

        return _matrix

    ########################################################################################
    ######## Helper methods for testing                                             ########
    ########################################################################################

    def assert_allclose(
        self, value: ArrayLike, target: ArrayLike, rtol: float = 1e-7, atol: float = 0.0
    ) -> None:  # pragma: no cover
        if isinstance(value, (CircuitResult, QuantumState)):
            value = value.state()
        if isinstance(target, (CircuitResult, QuantumState)):
            target = target.state()

        self.engine.testing.assert_close(value, target, rtol=rtol, atol=atol)

    ########################################################################################
    ######## Helper methods                                                         ########
    ########################################################################################

    def _cast_parameter(
        self, param: Union[ArrayLike, float, int], trainable: bool
    ) -> ArrayLike:
        """Cast a gate parameter to a torch tensor.

        Args:
            array (Union[int, float, complex]): Parameter to be casted.
            trainable (bool): If ``True``, the tensor requires gradient.
        """
        if isinstance(param, int) and trainable:
            return self.engine.tensor(
                param, dtype=self.parameter_dtype, requires_grad=True
            )
        if isinstance(param, float):
            return self.engine.tensor(
                param,
                dtype=self.parameter_dtype,
                requires_grad=trainable,
                device=self.device,
            )
        return self.engine.tensor(
            param, dtype=self.dtype, requires_grad=trainable, device=self.device
        )

    def _test_regressions(self, name: str):
        if name == "test_measurementresult_apply_bitflips":
            return [
                [0, 0, 0, 0, 2, 3, 0, 0, 0, 0],
                [0, 0, 0, 0, 2, 3, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 2, 0, 0, 0, 0, 0],
            ]

        if name == "test_probabilistic_measurement":
            if self.device == "cuda":  # pragma: no cover
                return {0: 273, 1: 233, 2: 242, 3: 252}
            return {1: 270, 2: 248, 3: 244, 0: 238}

        if name == "test_unbalanced_probabilistic_measurement":
            if self.device == "cuda":  # pragma: no cover
                return {0: 196, 1: 153, 2: 156, 3: 495}
            return {3: 492, 2: 176, 0: 168, 1: 164}

        return [
            {5: 30},
            {5: 17, 4: 5, 7: 4, 1: 2, 6: 2},
            {4: 9, 2: 5, 5: 5, 3: 4, 6: 4, 0: 1, 1: 1, 7: 1},
        ]
