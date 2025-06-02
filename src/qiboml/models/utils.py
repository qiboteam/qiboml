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
        self._mitigation_map_initial_popt = self.backend.cast(
            [1.0, 0.0], dtype="double"
        )
        self._mitigation_map_popt = self.backend.cast([1.0, 0.0], dtype="double")

        cfg = self.mitigation_config or {}
        self._real_time_mitigation = cfg.get("real_time", False)
        self._mitigation_method = cfg.get("method", "cdr")
        self._mitigation_method_kwargs = cfg.get("method_kwargs", {})

        custom_map = cfg.get("mitigation_kwargs", {}).get("model")
        if custom_map is not None:
            if not callable(custom_map):
                raise ValueError("Noise map model must be a callable")
            self._mitigation_map = custom_map
            defaults = custom_map.__defaults__ or ()
            self._mitigation_map_popt = self.backend.np.array(defaults)

    def data_regression(
        self,
        circuit: Circuit,
        observable: Union[ndarray, Hamiltonian],
        noise_model: NoiseModel,
        nshots: Optional[int],
    ):
        """Perform data regression on noisy and exact data."""
        _, _, popt, training_data = getattr(error_mitigation, self._mitigation_method)(
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

        # (example plotting)
        import matplotlib.pyplot as plt

        plt.scatter(training_data["noisy"], training_data["noise-free"])
        plt.savefig("cdr.pdf")
