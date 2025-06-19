from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

from qibo import Circuit, gates, transpiler
from qibo.backends import Backend, NumpyBackend, _check_backend
from qibo.config import log, raise_error
from qibo.hamiltonians import Hamiltonian, Z
from qibo.models.error_mitigation import error_sensitive_circuit
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
        noise_model (NoiseModel): a ``NoiseModel`` of Qibo, which is applied to the
            given circuit to perform noisy simulations. Default is ``None`` and
            no noise is added.
        density_matrix (bool): if ``True``, density matrix simulation is performed
            instead of state-vector simulation.
    """

    nqubits: int
    qubits: Optional[tuple[int]] = None
    wire_names: Optional[Union[tuple[int], Union[tuple[str]]]] = None
    nshots: Optional[int] = None
    backend: Optional[Backend] = None
    transpiler: Optional[Passes] = None
    noise_model: Optional[NoiseModel] = None
    density_matrix: Optional[bool] = False
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
        self._circuit = Circuit(
            self.nqubits, wire_names=wire_names, density_matrix=self.density_matrix
        )
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
        # Forcing the density matrix simulation if a noise model is given
        if self.noise_model is not None:
            density_matrix = True
        else:
            density_matrix = self.density_matrix
        # Aligning the density_matrix attribute of all the circuits
        self._circuit.init_kwargs["density_matrix"] = density_matrix
        x.init_kwargs["density_matrix"] = density_matrix

        wire_names = list(self.wire_names) if self.wire_names is not None else None
        x.wire_names = wire_names
        x.init_kwargs["wire_names"] = wire_names

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
            by default :math:`Z_0 + Z_1 + ... + Z_n` is used.
        mitigation_config (dict): configuration of the quantum error mitigation
            method in case it is desired. For example, one can set:
            ```
            mitigation_config = {
                "real_time": True,
                "threshold": 1e-1,
                "method": "CDR",
            }
    """

    observable: Union[ndarray, Hamiltonian] = None
    mitigation_config: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Ancillary post initialization operations."""
        if self.observable is None:
            self.observable = Z(self.nqubits, dense=True, backend=self.backend)

        # If mitigation is requested
        if self.mitigation_config is not None:
            # Construct the Mitigator object
            self.mitigator = Mitigator(
                mitigation_config=self.mitigation_config,
                backend=self.backend,
            )
            # This attribute will contain the reference circuit (Clifford and
            # error sensitive, that will be used in the real time error mitigation).
            self._mitigation_reference_circuit = None

        super().__post_init__()

    def __call__(self, x: Circuit) -> ndarray:
        """
        Execute the input circuit and calculate the expectation value of
        the internal observable on the final state.

        Args:
            x (Circuit): input Circuit.

        Returns:
            (ndarray): the calculated expectation value.
        """

        if self.mitigation_config is not None:
            self._check_or_update_mitigation_map(x)

        # run circuit
        if self.analytic:
            expval = self.observable.expectation(super().__call__(x).state())
        else:
            freqs = super().__call__(x).frequencies()
            expval = self.observable.expectation_from_samples(
                freqs, qubit_map=self.qubits
            )

        # apply mitigation if requested
        if self.mitigation_config is not None:
            expval = self._mitigated_expectation_value(expval)

        return expval.reshape(1, 1)

    def _mitigated_expectation_value(self, expval: float):
        """
        Compute the mitigated expectation value using the saved map and a given
        noisy expectation value.
        """
        mit_expval = self.backend.cast(
            self.mitigator._mitigation_map(
                expval, *self.mitigator._mitigation_map_popt
            ),
            dtype=self.backend.np.float64,
        )
        return mit_expval

    def _check_or_update_mitigation_map(self, x: Circuit):
        """
        If called for the first time, construct reference circuit and expectation
        value. If not, check whether a recomputation of the noise map is needed.
        The check is performed according to https://arxiv.org/abs/2311.05680.
        """
        if self._mitigation_reference_circuit is None:
            # Sample the reference error sensitive circuit (it will be pure Clifford)
            self._mitigation_reference_circuit = error_sensitive_circuit(
                circuit=x,
                observable=self.observable,
            )[0]
            # TODO: replace Numpy with Clifford after fixing Unitary problem
            simulation_backend = NumpyBackend()
            # Compute the reference expectation value once
            reference_frequencies = simulation_backend.execute_circuit(
                self._mitigation_reference_circuit,
                nshots=self.nshots,
            )
            self._mitigation_reference_value = self.observable.expectation_from_samples(
                reference_frequencies
            )

        # Compute the mitigated value of the reference circuit
        if self.analytic:
            expval = self.observable.expectation(
                super().__call__(self._mitigation_reference_circuit).state()
            )
        else:
            freqs = super().__call__(self._mitigation_reference_circuit).frequencies()
            expval = self.observable.expectation_from_samples(
                freqs, qubit_map=self.qubits
            )
        mit_expval = self._mitigated_expectation_value(expval)
        if (
            abs(mit_expval - self._mitigation_reference_value)
            > self.mitigator.threshold
        ):
            self.mitigator.data_regression(
                circuit=x + self._circuit,
                observable=self.observable,
                noise_model=self.noise_model,
                nshots=self.nshots,
            )

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
