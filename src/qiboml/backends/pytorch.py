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

    def __init__(self, dtype, device):
        import torch  # pylint: disable=import-outside-toplevel  # type: ignore

        super().__init__(dtype)
        self.engine = torch
        self.dtype = dtype
        self.device = device

    def _cast(self, array: ArrayLike, dtype, device: Optional[str] = None):
        if device is None:
            device = self.device

        flattened = [item for sublist in array for item in sublist]
        tensor_list = [
            self.engine.as_tensor(elem, dtype=dtype, device=device)
            for elem in flattened
        ]

        return self.engine.stack(tensor_list).reshape(len(array), len(array))

    def I(self, n: int = 2):
        return self.engine.eye(n, dtype=self.dtype, device=self.device)

    def Unitary(self, u: ArrayLike):
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
        dtype=None,
        copy: bool = False,
        device: Optional[str] = None,
    ) -> ArrayLike:
        """Casts input as a Torch tensor of the specified dtype.

        This method supports casting of single tensors or lists of tensors
        as for the :class:`qibo.backends.PyTorchBackend`.

        Args:
            array: Input to be casted.
            dtype (Union[str, torch.dtype, np.dtype, type]): Target data type.
                If ``None``, the default dtype of the backend is used.
                Defaults to ``None``.
            copy (bool, optional): If ``True``, the input tensor is copied before casting.
                Defaults to ``False``.
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
        if isinstance(array, self.engine.Tensor):
            return array.is_sparse

        return super().is_sparse(array)

    def set_device(self, device: str) -> None:  # pragma: no cover
        self.device = "cpu" if "CPU" in device else device

    def set_seed(self, seed) -> None:
        self.engine.manual_seed(seed)
        np.random.seed(seed)

    def set_threads(self, nthreads: int) -> None:  # pragma: no cover
        """Set number of threads for CPU backend simulations that accept it. Works in-place.

        Args:
            nthreads (int): Number of threads.
        """
        self.engine.set_num_threads(nthreads)

    def to_numpy(self, array: ArrayLike) -> ArrayLike:
        if isinstance(array, list):
            return np.asarray([self.to_numpy(i) for i in array])

        if isinstance(array, self.engine.Tensor):
            return array.cpu().numpy(force=True)

        return array

    ########################################################################################
    ######## Methods related to array manipulation                                  ########
    ########################################################################################

    def copy(self, array: ArrayLike, **kwargs) -> ArrayLike:
        return self.engine.clone(array, **kwargs)

    def default_rng(self, seed: Optional[int] = None):
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
        return self.engine.unsqueeze(array, axis)

    def expm(self, array: ArrayLike) -> ArrayLike:
        return self.engine.linalg.matrix_exp(array)  # pylint: disable=not-callable

    def flatnonzero(self, array: ArrayLike) -> ArrayLike:
        return self.engine.nonzero(array).flatten()

    def nonzero(self, array: ArrayLike) -> ArrayLike:
        rows_cols = self.engine.nonzero(array)
        return rows_cols[:, 0], rows_cols[:, 1]

    def random_choice(
        self,
        array: ArrayLike,
        size: Optional[Union[int, Tuple[int, ...]]] = None,
        replace: bool = True,
        p: ArrayLike = None,
        seed=None,
        **kwargs,
    ) -> ArrayLike:
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
        seed=None,
    ) -> ArrayLike:
        if high is None:
            high = low
            low = 0

        if size is None:
            size = (1,)
        elif isinstance(size, int):
            size = (size,)

        if seed is not None:
            local_state = self.default_rng(seed) if isinstance(seed, int) else seed

            return self.engine.randint(low, high, size, generator=local_state)

        return self.engine.randint(low, high, size)

    def random_normal(
        self,
        mean: Union[float, int],
        stddev: Union[float, int],
        size: Optional[Union[int, List[int], Tuple[int, ...]]] = None,
        seed=None,
        dtype: Optional[DTypeLike] = None,
    ) -> ArrayLike:
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

    def random_sample(self, size: Union[int, Tuple[int, ...]], seed=None) -> ArrayLike:
        if seed is not None:
            local_state = self.default_rng(seed) if isinstance(seed, int) else seed
        else:
            local_state = None

        return self.engine.rand(size, generator=local_state)

    def random_uniform(
        self,
        low: Union[float, int] = 0.0,
        high: Union[float, int] = 1.0,
        size: Optional[Union[int, Tuple[int, ...]]] = None,
        seed=None,
    ) -> ArrayLike:
        # if isinstance(size, int):
        #     size = (size,)

        if seed is not None:
            local_state = self.default_rng(seed) if isinstance(seed, int) else seed

            return low + (high - low) * self.engine.rand(size, generator=local_state)

        return low + (high - low) * self.engine.rand(size)

    def transpose(
        self, array: ArrayLike, axes: Union[Tuple[int, ...], List[int]] = None
    ) -> ArrayLike:
        return self.engine.permute(array, axes)

    def tril_indices(
        self, row: int, offset: int = 0, col: Optional[int] = None, **kwargs
    ) -> ArrayLike:
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
        parameters: ArrayLike = None,
        initial_state: ArrayLike = None,
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
        dtype=None,
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
        self, value, target, rtol: float = 1e-7, atol: float = 0.0
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

    def _order_probabilities(
        self, probs: ArrayLike, qubits: Tuple[int, ...], nqubits: int
    ) -> ArrayLike:
        """Arrange probabilities according to the given ``qubits`` ordering."""
        if probs.dim() == 0:  # pragma: no cover
            return probs
        unmeasured, reduced = [], {}
        for i in range(nqubits):
            if i in qubits:
                reduced[i] = i - len(unmeasured)
            else:
                unmeasured.append(i)
        return self.transpose(probs, [reduced.get(i) for i in qubits])

    def _test_regressions(self, name):
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
