from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Union

from qibo import Circuit
from qibo.backends import Backend, _check_backend
from qibo.config import log
from qibo.hamiltonians import Hamiltonian
from qibo.models import error_mitigation
from qibo.noise import NoiseModel

from qiboml import ndarray


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
        self._real_time_mitigation = cfg.get("real_time", False)
        self._mitigation_method = cfg.get("method", "cdr")
        self._mitigation_method_kwargs = cfg.get("method_kwargs", {})

        custom_map = cfg.get("method_kwargs", {}).get("model")
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

    def data_regression(
        self,
        circuit: Circuit,
        observable: Union[ndarray, Hamiltonian],
        noise_model: NoiseModel,
        nshots: Optional[int],
    ):
        """Perform data regression on noisy and exact data."""
        _, _, popt, _ = getattr(error_mitigation, self._mitigation_method)(
            circuit=circuit,
            observable=observable,
            noise_model=noise_model,
            nshots=nshots,
            full_output=True,
            **self._mitigation_method_kwargs,
        )

        self._mitigation_map.__defaults__ = tuple(popt)
        self._mitigation_map_popt = self.backend.np.array(popt)
        log.info(f"Obtained noise map params: {self._mitigation_map_popt}.")
