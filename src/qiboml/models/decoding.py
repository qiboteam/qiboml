from dataclasses import dataclass
from typing import Optional, Union

from qibo import Circuit, gates, transpiler
from qibo.backends import Backend, _check_backend
from qibo.config import raise_error
from qibo.hamiltonians import Hamiltonian, Z
from qibo.result import CircuitResult, MeasurementOutcomes, QuantumState
from qibo.transpiler import Passes
from qibo.quantum_info.metrics import infidelity

from qiboml import ndarray




@dataclass
class QuantumDecoding:
    """
    Abstract decoder class.

    Args:
        nqubits (int): total number of qubits.
        qubits (tuple[int], optional): set of qubits it acts on, by default ``range(nqubits)``.
        wire_names (tuple[int] | tuple[str], optional): names to be given to the wires, this has to
            have ``len`` equal to ``nqubits``. Additionally, this is mostly useful when executing
            on hardware to select which qubits to make use of. Namely, if the chip has qubits named:
            ```
            ("a", "b", "c", "d")
            ```
            and we wish to deploy a two qubits circuit on the first and last qubits you have to build it as:
            ```
            decoding = QuantumDecoding(nqubits=2, wire_names=("a", "d"))
            ```
        nshots (int, optional): number of shots used for circuit execution and sampling.
        backend (Backend, optional): backend used for computation, by default the globally-set backend is used.
        transpiler (Passes, optional): transpiler to run before circuit execution, by default no transpilation
                                       is performed on the circuit (``transpiler=None``).
    """

    nqubits: int
    qubits: Optional[tuple[int]] = None
    wire_names: Optional[Union[tuple[int], Union[tuple[str]]]] = None
    nshots: Optional[int] = None
    backend: Optional[Backend] = None
    transpiler: Optional[Passes] = None
    _circuit: Circuit = None

    def __post_init__(self):
        """Ancillary post initialization operations."""
        if self.qubits is None:
            self.qubits = tuple(range(self.nqubits))
        else:
            self.qubits = tuple(self.qubits)
        if self.wire_names is not None:
            # self.wire_names has to be a tuple to make the decoder hashable
            # and thus usable in Jax differentiation
            self.wire_names = tuple(self.wire_names)
        # I have to convert to list because qibo does not accept a tuple
        wire_names = list(self.wire_names) if self.wire_names is not None else None
        self._circuit = Circuit(self.nqubits, wire_names=wire_names)
        self.backend = _check_backend(self.backend)
        self._circuit.add(gates.M(*self.qubits))

    def __call__(
        self, x: Circuit
    ) -> Union[CircuitResult, QuantumState, MeasurementOutcomes]:
        """Combine the input circuir with the internal one and execute them with the internal backend.

        Args:
            x (Circuit): input circuit.

        Returns:
            (CircuitResult | QuantumState | MeasurementOutcomes): the execution ``qibo.result`` object.
        """
        self._circuit.density_matrix = x.density_matrix
        self._circuit.init_kwargs["density_matrix"] = x.density_matrix
        # same problem as above
        wire_names = list(self.wire_names) if self.wire_names is not None else None
        x.wire_names = wire_names
        x.init_kwargs["wire_names"] = wire_names

        if self.transpiler is not None:
            x, _ = self.transpiler(x)
        return self.backend.execute_circuit(x + self._circuit, nshots=self.nshots)

    @property
    def circuit(
        self,
    ) -> Circuit:
        """A copy of the internal circuit.

        Returns:
            (Circuit): a copy of the internal circuit.
        """
        return self._circuit.copy()

    def set_backend(self, backend: Backend):
        """Set the internal backend.

        Args:
            backend (Backend): backend to be set.
        """
        self.backend = backend

    @property
    def output_shape(self):
        """The shape of the decoded outputs."""
        raise_error(NotImplementedError)

    @property
    def analytic(self) -> bool:
        """Whether the decoder is analytic, i.e. the gradient is ananlytically computable, or not
        (e.g. if sampling is involved).

        Returns:
            (bool): ``True`` if ``nshots`` is ``None``, ``False`` otherwise.
        """
        if self.nshots is None:
            return True
        return False

    def __hash__(self) -> int:
        return hash((self.qubits, self.wire_names, self.nshots, self.backend))


class Probabilities(QuantumDecoding):
    """The probabilities decoder."""

    # TODO: collapse on ExpectationDecoding if not analytic

    def __call__(self, x: Circuit) -> ndarray:
        """Computes the final state probabilities.

        Args:
            x (Circuit): input circuit.

        Returns:
            (ndarray): the final probabilities.
        """
        return super().__call__(x).probabilities(self.qubits)

    @property
    def output_shape(self) -> tuple[int, int]:
        """Shape of the output probabilities.

        Returns:
            (tuple[int, int]): a ``(1, 2**nqubits)`` shape.
        """
        return (1, 2**self.nqubits)

    @property
    def analytic(self) -> bool:
        return True


@dataclass
class Expectation(QuantumDecoding):
    r"""The expectation value decoder.

    Args:
        observable (Hamiltonian | ndarray): the observable to calculate the expectation value of,
    by default :math:`Z_0\otimes Z_1\otimes ... \otimes Z_n` is used.
    """

    observable: Union[ndarray, Hamiltonian] = None

    def __post_init__(self):
        """Ancillary post initialization operations."""
        if self.observable is None:
            self.observable = Z(self.nqubits, dense=True, backend=self.backend)
        super().__post_init__()

    def __call__(self, x: Circuit) -> ndarray:
        """Execute the input circuit and calculate the expectation value of the internal observable on
        the final state

        Args:
            x (Circuit): input Circuit.

        Returns:
            (ndarray): the calculated expectation value.
        """
        if self.analytic:
            return self.observable.expectation(
                super().__call__(x).state(),
            ).reshape(1, 1)
        else:
            return self.observable.expectation_from_samples(
                super().__call__(x).frequencies(),
                qubit_map=self.qubits,
            ).reshape(1, 1)

    @property
    def output_shape(self) -> tuple[int, int]:
        """Shape of the output expectation value.

        Returns:
            (tuple[int, int]): a ``(1, 1)`` shape.
        """
        return (1, 1)

    def set_backend(self, backend: Backend):
        """Set the internal and observable's backends.

        Args:
            backend (Backend): backend to be set.
        """
        if isinstance(self.observable, Hamiltonian):
            matrix = self.backend.to_numpy(self.observable.matrix)
            super().set_backend(backend)
            self.observable = Hamiltonian(
                nqubits=self.nqubits,
                matrix=self.backend.cast(matrix),
                backend=self.backend,
            )
        else:
            super().set_backend(backend)
            self.observable.backend = backend

    def __hash__(self) -> int:
        return hash((self.qubits, self.nshots, self.backend, self.observable))


class State(QuantumDecoding):
    """The state decoder."""

    def __call__(self, x: Circuit) -> ndarray:
        """Compute the final state of the input circuit and separates it in its real and
        imaginary parts stacked on top of each other.

        Args:
            x (Circuit): input Circuit.

        Returns:
            (ndarray): the final state.
        """
        state = super().__call__(x).state()
        return self.backend.np.vstack(  # pylint: disable=no-member
            (
                self.backend.np.real(state),  # pylint: disable=no-member
                self.backend.np.imag(state),  # pylint: disable=no-member
            )
        ).reshape(self.output_shape)

    @property
    def output_shape(self) -> tuple[int, int, int]:
        """Shape of the output state.

        Returns:
            (tuple[int, int, int]): a ``(2, 1, 2**nqubits)`` shape.
        """
        return (2, 1, 2**self.nqubits)

    @property
    def analytic(self) -> bool:
        return True


class Samples(QuantumDecoding):
    """The samples decoder."""

    def __post_init__(self):
        super().__post_init__()

    def __call__(self, x: Circuit) -> ndarray:
        """Sample the final state of the circuit.

        Args:
            x (Circuit): input Circuit.

        Returns:
            (ndarray): the generated samples.
        """
        return self.backend.cast(super().__call__(x).samples(), self.backend.precision)

    @property
    def output_shape(self) -> tuple[int, int]:
        """Shape of the output samples.

        Returns:
            (tuple[int, int]): a ``(nshots, nqubits)`` shape.
        """
        return (self.nshots, len(self.qubits))

    @property
    def analytic(self) -> bool:  # pragma: no cover
        return False
    
@dataclass(kw_only=True)
class VariationalQuantumLinearSolver(QuantumDecoding):
    """Decoder for the Variational Quantum Linear Solver (VQLS).
    Adapted from the following paper arXiv:1909.05820v4 by Carlos Bravo-Prieto et al. 

    Args:
        target_state (ndarray): Target solution vector :math:`\\ket{b}`.
        A (ndarray): The matrix ``A`` in the linear system :math:`A \\, \\ket{x} = \\ket{b}`.
    
    Ensure QiboML backend set to Pytorch. 
    """
    target_state: ndarray
    A: ndarray

    def __post_init__(self):
        super().__post_init__()
        self.target_state = self.backend.cast(self.target_state, dtype=self.backend.np.complex128)
        self.A = self.backend.cast(self.A, dtype=self.backend.np.complex128)

        
    def __call__(self, circuit: Circuit):
        result = super().__call__(circuit)
        state = result.state()

        if self.A is None or self.target_state is None:
            raise_error(ValueError, "Both ``A`` and ``target_state`` must be provided.")  
        
        final_state = self.A @ state
        normalized = final_state / self.backend.calculate_vector_norm(final_state)
        cost = infidelity(normalized, self.target_state, backend=self.backend)
        return self.backend.cast(self.backend.np.real(cost), dtype=self.backend.np.float64)
    
    @property
    def output_shape(self) -> tuple[int, int]:
        return (1, 1)

    @property
    def analytic(self) -> bool:
        return True



