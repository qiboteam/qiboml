from typing import Union

from qibo.config import raise_error

from qiboml.backends.jax import JaxBackend
from qiboml.backends.pytorch import PyTorchBackend
from qiboml.backends.tensorflow import TensorflowBackend

PLATFORMS = ["tensorflow", "pytorch", "jax"]
AVAILABLE_PLATFORMS = [
    "tensorflow",
    "pytorch",
]  # temporary: to remove once pytorch and tensorflow are migrated  and jax is fully working
QibomlBackend = Union[TensorflowBackend, PyTorchBackend, JaxBackend]


class MetaBackend:
    """Meta-backend class which takes care of loading the qiboml backends."""

    @staticmethod
    def load(platform: str) -> QibomlBackend:
        """Load the qiboml backend.

        Args:
            platform (str): Name of the backend to load.
        Returns:
            qibo.backends.abstract.Backend: The loaded backend.
        """

        if platform == "tensorflow":
            return TensorflowBackend()
        elif platform == "pytorch":
            return PyTorchBackend()
        elif platform == "jax":
            return JaxBackend()
        else:
            raise_error(
                ValueError,
                f"Backend {platform} is not available. The qiboml backends are {PLATFORMS}.",
            )

    def list_available(self) -> dict:
        """List all the available qiboml backends."""
        available_backends = {}
        for platform in AVAILABLE_PLATFORMS:
            try:
                MetaBackend.load(platform)
                available = True
            except:  # pragma: no cover
                available = False
            available_backends[platform] = available
        return available_backends
