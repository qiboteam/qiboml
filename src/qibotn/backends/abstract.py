from abc import ABC

from qibo.config import raise_error


class QibotnBackend(ABC):

    def __init__(self):
        super().__init__()

    def apply_gate(self, gate, state, nqubits):  # pragma: no cover
        raise_error(NotImplementedError, "QiboTN cannot apply gates directly.")

    def apply_gate_density_matrix(self, gate, state, nqubits):  # pragma: no cover
        raise_error(NotImplementedError, "QiboTN cannot apply gates directly.")

    def assign_measurements(self, measurement_map, circuit_result):
        raise_error(NotImplementedError, "Not implemented in QiboTN.")

    def set_precision(self, precision):
        if precision != self.precision:
            super().set_precision(precision)
        self._setup_backend_specifics()

    def set_device(self, device):
        self.device = device
        self._setup_backend_specifics()

    def configure_tn_simulation(self, **config):
        """Configure the TN simulation that will be performed."""
        pass

    def _setup_backend_specifics(self):
        """Configure the backend specific according to the used package."""
        pass
