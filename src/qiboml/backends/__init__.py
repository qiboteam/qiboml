from qibo.backends.abstract import Backend
from qibo.config import raise_error

PLATFORMS = ("jax", "pytorch", "tensorflow")

# temporary: to remove once jax is fully working
AVAILABLE_PLATFORMS = ("pytorch", "tensorflow")


class MetaBackend:
    """Meta-backend class which takes care of loading the qiboml backends."""

    @staticmethod
    def load(platform: str) -> Backend:
        """Load the qiboml backend.

        Args:
            platform (str): Name of the backend to load.
        Returns:
            qibo.backends.abstract.Backend: The loaded backend.
        """

        if platform not in PLATFORMS:
            raise_error(
                ValueError,
                f"Backend {platform} is not available. The qiboml backends are {PLATFORMS}.",
            )

        if platform == "tensorflow":
            from qiboml.backends.tensorflow import (  # pylint: disable=C0415
                TensorflowBackend,
            )

            return TensorflowBackend()

        if platform == "pytorch":
            from qiboml.backends.pytorch import PyTorchBackend  # pylint: disable=C0415

            return PyTorchBackend()

        from qiboml.backends.jax import JaxBackend  # pylint: disable=C0415

        return JaxBackend()

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
