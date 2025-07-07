from contextlib import contextmanager
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
from qibo.quantum_info.metrics import infidelity

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
            given circuit to perform noisy simulations. In case a `transpiler` is
            passed, the noise model is applied to the transpiled circuit.
            Default is ``None`` and no noise is added.
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

    @contextmanager
    def _temporary_nshots(self, nshots):
        """Context manager to execute the decoder with a custom number of shots."""
        original = self.nshots
        self.nshots = nshots
        try:
            yield
        finally:
            self.nshots = original

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
        mitigation_config (dict): configuration of the real-time quantum error mitigation
            method in case it is desired.
            The real-time quantum error mitigation algorithm is proposed in https://arxiv.org/abs/2311.05680
            and consists in performing a real-time check of the reliability of a learned mitigation map.
            This is done by constructing a reference error-sensitive Clifford circuit,
            which preserves the size of the original, target one. When the decoder is called,
            the reliability of the mitigation map is checked by computing
            a simple metric :math:`D = |E_{\rm noisy} - E_{\rm mitigated}|`. If
            the metric is found exceeding an arbitrary threshold value :math:`\delta`,
            then a chosen data-driven error mitigation technique is executed to
            retrieve the mitigation map.
            To successfully check the reliability of the mitigation map or computing
            the map itself, it is recommended to use a number of shots which leads
            to a statistical noise (due to measurements) :math:`\varepsilon << \delta`.
            For this reason, the real-time error mitigation algorithm can be customized
            by passing also a `min_iterations` argument, which will define the minimum
            number of decoding calls which have to happen before the mitigation map
            check is performed.
            An example of real-time error mitigation configuration is:

            .. code-block:: python

                mitigation_config = {
                    "threshold": 2e-1,
                    "min_iterations": 500,
                    "method": "CDR",
                    "method_kwargs": {"n_training_samples": 100, "nshots": 10000},
                }

            The given example is performing real-time error mitigation with the
            request of computing the mitigation map via Clifford Data Regression
            whenever the reference expectation value differs from the mitigated
            one of :math:`\delta > 0.2`. This check is performed every 500 iterations and,
            in case it is required, the mitigation map is computed executing circuits
            with `nshots=10000`.
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
            _real_time_mitigation_check(self, x)

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
            expval = self.backend.cast(
                self.mitigator(expval),
                dtype=self.backend.np.float64,
            )

        return expval.reshape(1, 1)

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
    target_state: ndarray = None
    A: ndarray = None

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






def _real_time_mitigation_check(decoder: Expectation, x: Circuit):
    """
    Helper function to execute the real time mitigation check
    and, if necessary, to compute the reference circuit expectation value.
    """
    # At first iteration, compute the reference value (exact)
    if decoder.mitigator._reference_value is None:
        decoder.mitigator.calculate_reference_expval(
            observable=decoder.observable,
            circuit=x,
        )
        # Trigger the mechanism at first iteration
        _check_or_recompute_map(decoder, x)

    if decoder.mitigator._iteration_counter == decoder.mitigator._min_iterations:
        log.info("Checking map since max iterations reached.")
        _check_or_recompute_map(decoder, x)
        decoder.mitigator._iteration_counter = 0
    else:
        decoder.mitigator._iteration_counter += 1


def _check_or_recompute_map(decoder: Expectation, x: Circuit):
    """Helper function to recompute the mitigation map."""
    # Compute the expectation value of the reference circuit
    with decoder._temporary_nshots(decoder.mitigator._nshots):
        freqs = (
            super(Expectation, decoder)
            .__call__(decoder.mitigator._reference_circuit)
            .frequencies()
        )
        reference_expval = decoder.observable.expectation_from_samples(
            freqs, qubit_map=decoder.qubits
        )
    # Check or update noise map
    decoder.mitigator.check_or_update_map(
        noisy_reference_value=reference_expval,
        circuit=x + decoder._circuit,
        observable=decoder.observable,
        noise_model=decoder.noise_model,
        nshots=decoder.nshots,
    )

