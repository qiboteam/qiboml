"""PyTorch backend."""

from typing import Union

import numpy as np
import qibo.backends.einsum_utils as einsum_utils
from qibo import Circuit, __version__, gates
from qibo.backends.npmatrices import NumpyMatrices
from qibo.backends.numpy import NumpyBackend
from qibo.result import CircuitResult, MeasurementOutcomes, QuantumState

import qiboml.backends.einsum_utils as batched_einsum_utils
from qiboml.result import BatchedResult


class TorchMatrices(NumpyMatrices):
    """Matrix representation of every gate as a torch Tensor.

    Args:
        dtype (torch.dtype): Data type of the matrices.
    """

    def __init__(self, dtype, device):
        import torch  # pylint: disable=import-outside-toplevel  # type: ignore

        super().__init__(dtype)
        self.np = torch
        self.dtype = dtype
        self.device = device

    def _cast(self, x, dtype, device=None):
        if device is None:
            device = self.device
        flattened = [item for sublist in x for item in sublist]
        tensor_list = [
            self.np.as_tensor(i, dtype=dtype, device=device) for i in flattened
        ]
        return self.np.stack(tensor_list).reshape(len(x), len(x))

    def I(self, n=2):
        return self.np.eye(n, dtype=self.dtype, device=self.device)

    def Unitary(self, u):
        return self._cast(u, dtype=self.dtype, device=self.device)


class PyTorchBackend(NumpyBackend):
    def __init__(self):
        super().__init__()
        import torch  # pylint: disable=import-outside-toplevel  # type: ignore

        self.np = torch

        self.name = "qiboml"
        self.platform = "pytorch"

        self.versions = {
            "qibo": __version__,
            "numpy": np.__version__,
            "torch": self.np.__version__,
        }

        # Default data type used for the gate matrices is complex128
        self.dtype = self._torch_dtype(self.dtype)
        # Default data type used for the real gate parameters is float64
        self.parameter_dtype = self._torch_dtype("float64")
        self.device = self.np.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.matrices = TorchMatrices(self.dtype, self.device)
        self.nthreads = 0
        self.tensor_types = (self.np.Tensor, np.ndarray)

        # These functions in Torch works in a different way than numpy or have different names
        self.np.transpose = self.np.permute
        self.np.copy = self.np.clone
        self.np.power = self.np.pow
        self.np.expand_dims = self.np.unsqueeze
        self.np.mod = self.np.remainder
        self.np.right_shift = self.np.bitwise_right_shift
        self.np.sign = self.np.sgn
        self.np.flatnonzero = lambda x: self.np.nonzero(x).flatten()

        # These functions are device dependent
        torch_zeros = self.np.zeros

        def zeros(shape, dtype=None, device=None):
            if dtype is None:
                dtype = self.dtype
            if device is None:
                device = self.device
            return torch_zeros(shape, dtype=dtype, device=device)

        setattr(self.np, "zeros", zeros)

    def _torch_dtype(self, dtype):
        if dtype == "float":
            dtype += "32"
        return getattr(self.np, dtype)

    def set_device(self, device):  # pragma: no cover
        self.device = device

    def cast(
        self,
        x,
        dtype=None,
        copy: bool = False,
        device=None,
    ):
        """Casts input as a Torch tensor of the specified dtype.

        This method supports casting of single tensors or lists of tensors
        as for the :class:`qibo.backends.PyTorchBackend`.

        Args:
            x (Union[torch.Tensor, list[torch.Tensor], np.ndarray, list[np.ndarray], int, float, complex]):
                Input to be casted.
            dtype (Union[str, torch.dtype, np.dtype, type]): Target data type.
                If ``None``, the default dtype of the backend is used.
                Defaults to ``None``.
            copy (bool, optional): If ``True``, the input tensor is copied before casting.
                Defaults to ``False``.
        """

        if dtype is None:
            dtype = self.dtype
        elif isinstance(dtype, type):
            dtype = self._torch_dtype(dtype.__name__)
        elif not isinstance(dtype, self.np.dtype):
            dtype = self._torch_dtype(str(dtype))

        if device is None:
            device = self.device

        if isinstance(x, self.np.Tensor):
            x = x.to(dtype)
        elif (
            isinstance(x, list)
            and len(x) > 0
            and all(isinstance(row, self.np.Tensor) for row in x)
        ):
            x = self.np.stack(x)
        else:
            x = self.np.tensor(x, dtype=dtype)

        if copy:
            return x.clone().to(device)
        return x.to(device)

    def matrix_parametrized(self, gate):
        """Convert a parametrized gate to its matrix representation in the computational basis."""
        name = gate.__class__.__name__
        _matrix = getattr(self.matrices, name)
        if name == "GeneralizedRBS":
            for parameter in ["theta", "phi"]:
                if not isinstance(gate.init_kwargs[parameter], self.np.Tensor):
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
        else:
            new_parameters = []
            for parameter in gate.parameters:
                if not isinstance(parameter, self.np.Tensor):
                    parameter = self._cast_parameter(
                        parameter, trainable=gate.trainable
                    )
                elif parameter.requires_grad:
                    gate.trainable = True
                new_parameters.append(parameter)
            gate.parameters = tuple(new_parameters)
        _matrix = _matrix(*gate.parameters)
        return _matrix

    def _cast_parameter(self, x, trainable):
        """Cast a gate parameter to a torch tensor.

        Args:
            x (Union[int, float, complex]): Parameter to be casted.
            trainable (bool): If ``True``, the tensor requires gradient.
        """
        if isinstance(x, int) and trainable:
            return self.np.tensor(x, dtype=self.parameter_dtype, requires_grad=True)
        if isinstance(x, float):
            return self.np.tensor(
                x,
                dtype=self.parameter_dtype,
                requires_grad=trainable,
                device=self.device,
            )
        return self.np.tensor(
            x, dtype=self.dtype, requires_grad=trainable, device=self.device
        )

    def is_sparse(self, x):
        if isinstance(x, self.np.Tensor):
            return x.is_sparse

        return super().is_sparse(x)

    def to_numpy(self, x):
        if isinstance(x, list):
            return np.asarray([self.to_numpy(i) for i in x])

        if isinstance(x, self.np.Tensor):
            return x.cpu().numpy(force=True)

        return x

    def _order_probabilities(self, probs, qubits, nqubits):
        """Arrange probabilities according to the given ``qubits`` ordering."""
        if probs.dim() == 0:  # pragma: no cover
            return probs
        unmeasured, reduced = [], {}
        for i in range(nqubits):
            if i in qubits:
                reduced[i] = i - len(unmeasured)
            else:
                unmeasured.append(i)
        return self.np.transpose(probs, [reduced.get(i) for i in qubits])

    def calculate_probabilities(self, state, qubits, nqubits):
        rtype = self.np.real(state).dtype
        unmeasured_qubits = tuple(i for i in range(nqubits) if i not in qubits)
        state = self.np.reshape(self.np.abs(state) ** 2, nqubits * (2,))
        if len(unmeasured_qubits) == 0:
            probs = self.cast(state, dtype=rtype)
        else:
            probs = self.np.sum(self.cast(state, dtype=rtype), axis=unmeasured_qubits)
        return self._order_probabilities(probs, qubits, nqubits).ravel()

    def set_seed(self, seed):
        self.np.manual_seed(seed)
        np.random.seed(seed)

    def sample_shots(self, probabilities, nshots):
        return self.np.multinomial(
            self.cast(probabilities, dtype="float"), nshots, replacement=True
        )

    def calculate_eigenvalues(self, matrix, k: int = 6, hermitian: bool = True):
        if hermitian:
            return self.np.linalg.eigvalsh(matrix)  # pylint: disable=not-callable
        return self.np.linalg.eigvals(matrix)  # pylint: disable=not-callable

    def calculate_eigenvectors(self, matrix, k: int = 6, hermitian: int = True):
        if hermitian:
            return self.np.linalg.eigh(matrix)  # pylint: disable=not-callable
        return self.np.linalg.eig(matrix)  # pylint: disable=not-callable

    def calculate_matrix_exp(self, a, matrix, eigenvectors=None, eigenvalues=None):
        if eigenvectors is None or self.is_sparse(matrix):
            return self.np.linalg.matrix_exp(  # pylint: disable=not-callable
                -1j * a * matrix
            )
        expd = self.np.diag(self.np.exp(-1j * a * eigenvalues))
        ud = self.np.conj(eigenvectors).T
        return self.np.matmul(eigenvectors, self.np.matmul(expd, ud))

    def calculate_matrix_power(
        self,
        matrix,
        power: Union[float, int],
        precision_singularity: float = 1e-14,
    ):
        copied = self.cast(matrix, copy=True)
        copied = self.to_numpy(copied) if power >= 0.0 else copied.detach()
        copied = super().calculate_matrix_power(copied, power, precision_singularity)
        return self.cast(copied, dtype=copied.dtype)

    def calculate_jacobian_matrix(
        self, circuit, parameters=None, initial_state=None, return_complex: bool = True
    ):
        copied = circuit.copy(deep=True)

        def func(parameters):
            """torch requires object(s) to be wrapped in a function."""
            copied.set_parameters(parameters)
            state = self.execute_circuit(copied, initial_state=initial_state).state()
            if return_complex:
                return self.np.real(state), self.np.imag(state)
            return self.np.real(state)

        return self.np.autograd.functional.jacobian(func, parameters)

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

        if name == "test_post_measurement_bitflips_on_circuit":
            return [
                {5: 30},
                {5: 17, 4: 5, 7: 4, 1: 2, 6: 2},
                {4: 9, 2: 5, 5: 5, 3: 4, 6: 4, 0: 1, 1: 1, 7: 1},
            ]

    def apply_gate_batched(self, gate, state, nqubits):
        state = self.np.reshape(state, (state.shape[0],) + nqubits * (2,))
        matrix = gate.matrix(self)
        if gate.is_controlled_by:
            matrix = self.np.reshape(matrix, 2 * len(gate.target_qubits) * (2,))
            ncontrol = len(gate.control_qubits)
            nactive = nqubits - ncontrol
            order, targets = batched_einsum_utils.control_order(
                gate.control_qubits, nqubits
            )
            state = self.np.transpose(state, order)
            # Apply `einsum` only to the part of the state where all controls
            # are active. This should be `state[-1]`
            state = self.np.reshape(
                state, (state.shape[0],) + (2**ncontrol,) + nactive * (2,)
            )
            opstring = batched_einsum_utils.apply_gate_string(targets, nactive)
            updates = self.np.einsum(opstring, state[:, -1], matrix)
            # Concatenate the updated part of the state `updates` with the
            # part of of the state that remained unaffected `state[:-1]`.
            state = self.np.concatenate([state[:, :-1], updates[None]], axis=1)
            state = self.np.reshape(state, (state.shape[0],) + nqubits * (2,))
            # Put qubit indices back to their proper places
            state = self.np.transpose(state, batched_einsum_utils.reverse_order(order))
        else:
            matrix = self.np.reshape(matrix, 2 * len(gate.qubits) * (2,))
            opstring = batched_einsum_utils.apply_gate_string(gate.qubits, nqubits)
            state = self.np.einsum(opstring, state, matrix)
        return self.np.reshape(state, (state.shape[0], -1) + (2**nqubits,))

    def apply_gates_same_qubits_batched(self, gate_list, state, nqubits):
        qubits = gate_list[0].qubits
        state = self.np.reshape(state, (state.shape[0],) + nqubits * (2,))
        matrix = self.np.reshape(
            self.np.vstack([g.matrix(self) for g in gate_list]),
            (len(gate_list),) + 2 * len(gate_list[0].qubits) * (2,),
        )
        if False:  # gate.is_controlled_by:
            raise NotImplementedError
        else:
            opstring = batched_einsum_utils.apply_gates_same_qubits_string(
                qubits, nqubits
            )
            state = self.np.einsum(opstring, state, matrix)
        return self.np.reshape(state, (state.shape[0], -1) + (2**nqubits,))

    def execute_batch_of_circuits(
        self, circuits: list[Circuit], initial_state=None, nshots: int = 1000
    ):

        try:
            nqubits = circuits[0].nqubits

            if circuits[0].density_matrix:
                if initial_state is None:
                    state = self.zero_density_matrix(nqubits)
                else:
                    state = self.cast(initial_state)

                for i in range(len(circuits[0].queue)):
                    _gates = [c.queue[i] for c in circuits]

                    state = gate.apply_density_matrix_batched(self, state, nqubits)

            else:
                if initial_state is None:
                    state = self.zero_state(nqubits)
                    state = self.np.vstack(
                        len(circuits) * (state.reshape(1, -1, 2**nqubits),)
                    )
                else:
                    state = self.cast(initial_state)

                for i in range(len(circuits[0].queue)):
                    _gates = [c.queue[i] for c in circuits]
                    first_gate = _gates[0]
                    same_gate = all(isinstance(g, first_gate.__class__) for g in _gates)
                    same_qubits = all(g.qubits == first_gate.qubits for g in _gates)
                    par_gate = isinstance(first_gate, gates.ParametrizedGate)
                    if same_qubits:
                        if same_gate and not par_gate:
                            state = self.apply_gate_batched(first_gate, state, nqubits)
                        else:
                            state = self.apply_gates_same_qubits_batched(
                                _gates, state, nqubits
                            )
                    else:
                        raise NotImplementedError

            circuit = circuits[0]
            if circuit.has_unitary_channel:
                # here we necessarily have `density_matrix=True`, otherwise
                # execute_circuit_repeated would have been called
                if circuit.measurements:
                    circuit._final_state = BatchedResult(
                        [
                            CircuitResult(
                                s, c.measurements, backend=self, nshots=nshots
                            )
                            for s, c in zip(state, circuits)
                        ]
                    )
                    return circuit._final_state
                else:
                    circuit._final_state = BatchedResult(
                        [QuantumState(s, backend=self) for s in state]
                    )
                    return circuit._final_state

            else:
                if circuit.measurements:
                    circuit._final_state = BatchedResult(
                        [
                            CircuitResult(
                                s, c.measurements, backend=self, nshots=nshots
                            )
                            for s, c in zip(state, circuits)
                        ]
                    )
                    return circuit._final_state
                else:
                    circuit._final_state = BatchedResult(
                        [QuantumState(s, backend=self) for s in state]
                    )
                    return circuit._final_state
        except:
            pass
