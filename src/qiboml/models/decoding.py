from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

from qibo import Circuit, gates, transpiler
from qibo.backends import Backend, _check_backend
from qibo.config import log, raise_error
from qibo.hamiltonians import Hamiltonian, Z
from qibo.noise import NoiseModel
from qibo.result import CircuitResult, MeasurementOutcomes, QuantumState
from qibo.transpiler import Passes

from qiboml import ndarray
from qiboml.models.utils import Mitigator


@dataclass
class QuantumDecoding:
    """
    Abstract decoder class.

    Args:
        nqubits (int): total number of qubits.
        qubits (tuple[int], optional): set of qubits it acts on, by default ``range(nqubits)``.
        nshots (int, optional): number of shots used for circuit execution and sampling.
        backend (Backend, optional): backend used for computation, by default the globally-set backend is used.
        transpiler (Passes, optional): transpiler to run before circuit execution, by default no transpilation
                                       is performed on the circuit (``transpiler=None``).
    """

    nqubits: int
    qubits: Optional[tuple[int]] = None
    nshots: Optional[int] = None
    backend: Optional[Backend] = None
    transpiler: Optional[Passes] = None
    noise_model: Optional[NoiseModel] = None
    density_matrix: Optional[bool] = False
    _circuit: Circuit = None

    def __post_init__(self):
        """Ancillary post initialization operations."""
        self.qubits = (
            tuple(range(self.nqubits)) if self.qubits is None else tuple(self.qubits)
        )
        self._circuit = Circuit(self.nqubits, density_matrix=self.density_matrix)
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
        # Forcing the density matrix simulation if a noise model is given
        if self.noise_model is not None:
            self._circuit.init_kwargs["density_matrix"] = True
        else:
            self._circuit.init_kwargs["density_matrix"] = x.density_matrix

        if self.transpiler is not None:
            x, _ = self.transpiler(x)

        if self.noise_model is not None:
            executable_circuit = self.noise_model.apply(x + self._circuit)
        else:
            executable_circuit = x + self._circuit

        return self.backend.execute_circuit(executable_circuit, nshots=self.nshots)

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
        return hash((self.qubits, self.nshots, self.backend))


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
    """The expectation value decoder."""

    observable: Union[ndarray, Hamiltonian] = None
    mitigation_config: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.observable is None:
            self.observable = Z(self.nqubits, dense=True, backend=self.backend)

        # ensure config dict
        if self.mitigation_config is None:
            self.mitigation_config = {}

        # Construct the Mitigator object
        self.mitigator = Mitigator(
            mitigation_config=self.mitigation_config,
            backend=self.backend,
        )

        super().__post_init__()

    def __call__(self, x: Circuit) -> ndarray:
        # recompute map if real-time enabled and not yet run
        if self.mitigator._real_time_mitigation and self.backend.np.allclose(
            self.mitigator._mitigation_map_popt,
            self.mitigator._mitigation_map_initial_popt,
            atol=1e-6,
        ):
            self.mitigator.data_regression(
                circuit=x + self._circuit,
                observable=self.observable,
                noise_model=self.noise_model,
                nshots=self.nshots,
            )

        # run circuit
        if self.analytic:
            expval = self.observable.expectation(super().__call__(x).state())
        else:
            freqs = super().__call__(x).frequencies()
            expval = self.observable.expectation_from_samples(
                freqs, qubit_map=self.qubits
            )

        # apply mitigation
        return self.backend.cast(
            self.backend.to_numpy(
                self.mitigator._mitigation_map(
                    expval, *self.mitigator._mitigation_map_popt
                )
            ),
            dtype="double",
        ).reshape(1, 1)

    @property
    def output_shape(self) -> tuple[int, int]:
        return (1, 1)

    def set_backend(self, backend: Backend):
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
