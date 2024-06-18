from typing import Union

from qibo.config import raise_error

from qiboml.backends.jax import JaxBackend
from qiboml.backends.pytorch import PyTorchBackend
from qiboml.backends.tensorflow import TensorflowBackend

QIBOML_BACKENDS = ["tensorflow", "pytorch", "jax"]
QibomlBackend = Union[TensorflowBackend, PyTorchBackend, JaxBackend]


class MetaBackend:
    """Meta-backend class which takes care of loading the qiboml backends."""

    @staticmethod
    def load(backend: str, **kwargs) -> QibomlBackend:
        """Loads the qiboml backend.

        Args:
            backend (str): Name of the backend to load.
            kwargs (dict): Additional arguments for the qibo backend.
        Returns:
            qibo.backends.abstract.Backend: The loaded backend.
        """

        if backend == "tensorflow":
            return TensorflowBackend()
        elif backend == "pytorch":
            return PyTorchBackend()
        elif backend == "jax":
            return JaxBackend()
        else:
            raise_error(
                ValueError,
                f"Backend {backend} is not available. The qiboml backends are {QIBOML_BACKENDS}.",
            )

    def list_available(self) -> dict:
        """Lists all the available native qibo backends."""
        available_backends = {}
        for backend in QIBOML_BACKENDS:
            try:
                MetaBackend.load(backend)
                available = True
            except:  # pragma: no cover
                available = False
            available_backends[backend] = available
        return available_backends
