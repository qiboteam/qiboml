from typing import Union

from qibo.config import raise_error

from qibotn.backends.abstract import QibotnBackend
from qibotn.backends.cutensornet import CuTensorNet  # pylint: disable=E0401
from qibotn.backends.quimb import QuimbBackend  # pylint: disable=E0401

PLATFORMS = ("cutensornet", "quimb", "qmatchatea")


class MetaBackend:
    """Meta-backend class which takes care of loading the qibotn backends."""

    @staticmethod
    def load(platform: str, runcard: dict = None) -> QibotnBackend:
        """Loads the backend.

        Args:
            platform (str): Name of the backend to load: either `cutensornet`, 'qmatchatea' or `quimb`.
            runcard (dict): Dictionary containing the simulation settings.
        Returns:
            qibo.backends.abstract.Backend: The loaded backend.
        """

        if platform == "cutensornet":  # pragma: no cover
            return CuTensorNet(runcard)
        elif platform == "quimb":  # pragma: no cover
            from qibotn.backends.quimb import QuimbBackend

            return QuimbBackend()
        elif platform == "qmatchatea":  # pragma: no cover
            from qibotn.backends.qmatchatea import QMatchaTeaBackend

            return QMatchaTeaBackend()
        else:
            raise_error(
                NotImplementedError,
                f"Unsupported platform {platform}, please pick one in {PLATFORMS}",
            )

    def list_available(self) -> dict:
        """Lists all the available qibotn backends."""
        available_backends = {}
        for platform in PLATFORMS:
            try:
                MetaBackend.load(platform=platform)
                available = True
            except:
                available = False
            available_backends[platform] = available
        return available_backends
