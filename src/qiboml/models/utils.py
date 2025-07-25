from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Union

from qibo import Circuit
from qibo.backends import Backend, NumpyBackend, _check_backend
from qibo.config import log
from qibo.hamiltonians import Hamiltonian
from qibo.models import error_mitigation
from qibo.noise import NoiseModel

from qiboml import ndarray

REF_CIRCUIT_SEED = 42


@dataclass
class Mitigator:
    """
    The ``Mitigator`` object keep track of the mitigation procedure and implements
    the necessary operations to perform real-time quantum error mitigation using
    data-driven approaches.

    Args:
        mitigation_config (Optional[Dict]): configuration of the chosen error mitigation
            strategy.
    """

    mitigation_config: Optional[Dict[str, Any]] = None
    backend: Optional[Backend] = None

    def __post_init__(self):

        self.backend = _check_backend(self.backend)
        self._mitigation_map: Callable[..., ndarray] = lambda x, a=1, b=0: a * x + b

        cfg = self.mitigation_config or {}
        self._threshold = cfg.get("threshold", 1e-1)
        self._min_iterations = cfg.get("min_iterations", 100)
        self._iteration_counter = 0
        self._mitigation_method = cfg.get("method", "cdr")
        self._mitigation_method_kwargs = cfg.get("method_kwargs", {})
        self._nshots = self._mitigation_method_kwargs.get("nshots", 10000)

        custom_map = self._mitigation_method_kwargs.get("model")
        if custom_map is not None:
            if not callable(custom_map):
                raise ValueError("Noise map model must be a callable")
            self._mitigation_map = custom_map

        n_params = self._mitigation_map.__code__.co_argcount - 1
        defaults = self._mitigation_map.__defaults__ or tuple(
            [1.0] * (n_params - 1) + [0.0]
        )
        self._mitigation_map_initial_popt = self.backend.cast(defaults, dtype="double")
        self._mitigation_map_popt = self.backend.cast(defaults, dtype="double")
        self._mitigation_function = getattr(error_mitigation, self._mitigation_method)
        # TODO: replace with Clifford backend once the Unitary bug is fixed
        self._simulation_backend = NumpyBackend()
        self._reference_circuit = None
        self._reference_value = None

        self._n_checks = 0
        self._n_maps_computed = 0

    def __call__(self, expval):
        """
        Take a noisy expectation value and return value mitigated with current
        available map.
        """
        return self._mitigation_map(expval, *self._mitigation_map_popt)

    def calculate_reference_expval(
        self,
        observable: Union[ndarray, Hamiltonian],
        circuit: Circuit,
    ):
        """Construct reference error sensitive circuit."""
        self._reference_circuit = error_mitigation.error_sensitive_circuit(
            circuit=circuit, observable=observable
        )[0]
        # Execute the reference circuit
        reference_state = self._simulation_backend.execute_circuit(
            self._reference_circuit,
        ).state()
        self._reference_value = observable.expectation(reference_state)

    def map_is_reliable(self, noisy_reference_value: ndarray):
        """
        Check the distance between the reference value and the mitigated one.

        Args:
            noisy_reference_value (ndarray): the noisy expectation value obtained
                executing the reference circuit on a noisy engine.

        Returns:
            bool: ``True`` if the map is reliable, ``False`` if not.
        """
        mitigated_ref_value = self.__call__(noisy_reference_value)
        if abs(mitigated_ref_value - self._reference_value) > self._threshold:
            return False
        return True

    def check_or_update_map(
        self,
        noisy_reference_value: ndarray,
        circuit: Circuit,
        observable: Union[ndarray, Hamiltonian],
        noise_model: NoiseModel,
    ):
        """
        Check if the mitigation map is reliable. If not, execute the
        error mitigation technique and recompute it.
        """
        if not self.map_is_reliable(noisy_reference_value):
            self.data_regression(
                circuit=circuit,
                observable=observable,
                noise_model=noise_model,
            )
            self._n_maps_computed += 1
        self._n_checks += 1

    def data_regression(
        self,
        circuit: Circuit,
        observable: Union[ndarray, Hamiltonian],
        noise_model: NoiseModel,
    ):
        """
        Perform data regression on noisy and exact data.
        """

        _, _, popt, _ = self._mitigation_function(
            circuit=circuit,
            observable=observable,
            noise_model=noise_model,
            full_output=True,
            **self._mitigation_method_kwargs,
        )

        self._mitigation_map.__defaults__ = tuple(popt)
        self._mitigation_map_popt = self.backend.cast(popt, dtype="double")
        log.info(f"Obtained noise map params: {self._mitigation_map_popt}.")


def _get_wire_names_and_qubits(nqubits, qubits):

    if qubits is not None:
        if isinstance(qubits[0], str):
            wire_names = qubits
            qubits = tuple(range(len(qubits)))
        else:
            qubits = tuple(qubits)
            wire_names = None
    else:
        qubits = tuple(range(nqubits))
        wire_names = None
    return qubits, wire_names
