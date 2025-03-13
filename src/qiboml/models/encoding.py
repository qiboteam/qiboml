from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np
import tensorflow as tf
import tensorflow.experimental.numpy as tnp
from qibo import Circuit, gates
from qibo.config import raise_error

from qiboml import ndarray


@dataclass
class QuantumEncoding(ABC):
    """
    Abstract Encoder class.

    Args:
        nqubits (int): total number of qubits.
        qubits (tuple[int], optional): set of qubits it acts on, by default ``range(nqubits)``.
    """

    nqubits: int
    qubits: Optional[tuple[int]] = None
    _circuit: Circuit = None

    def __post_init__(
        self,
    ):
        """Ancillary post initialization for the dataclass object."""
        self.qubits = (
            tuple(range(self.nqubits)) if self.qubits is None else tuple(self.qubits)
        )

        self._circuit = Circuit(self.nqubits)

    @abstractmethod
    def __call__(self, x: ndarray) -> Circuit:
        """Abstract call method."""
        pass

    @property
    def circuit(
        self,
    ) -> Circuit:
        """Internal initialized circuit."""
        return self._circuit.copy()

    @property
    def differentiable(self) -> bool:
        """Whether the encoder is differentiable. If ``True`` the gradient w.r.t. the inputs is
        calculated, otherwise it is automatically set to zero.
        """
        return True

    def __hash__(self) -> int:
        return hash(self.qubits)


class PhaseEncoding(QuantumEncoding):

    def ___post_init__(
        self,
    ):
        """Ancillary post initialization: builds the internal circuit with the rotation gates."""
        super().__post_init__()
        for q in self.qubits:
            self._circuit.add(gates.RY(q, theta=0.0, trainable=False))

    def _set_phases(self, x: ndarray):
        """Helper method to set the phases of the rotations of the internal circuit.

        Args:
            x (ndarray): the input rotation angles.
        """
        for gate, phase in zip(self._circuit.parametrized_gates, x.ravel()):
            gate.parameters = phase

    def __call__(self, x: ndarray) -> Circuit:
        """Construct the circuit encoding the ``x`` data in the rotation angles of some
        ``RY`` gates.

        Args:
            x (ndarray): the input real data to encode in rotation angles.

        Returns:
            (Circuit): the constructed ``qibo.Circuit``.
        """
        circuit = self.circuit
        x = x.ravel()
        for i, q in enumerate(self.qubits):
            circuit.add(gates.RY(q, theta=x[i], trainable=False))
        # self._set_phases(x)
        return circuit


class BinaryEncoding(QuantumEncoding):

    def __call__(self, x: ndarray) -> Circuit:
        r"""Construct the circuit encoding the ``x`` binary data in some ``RX`` rotation gates
        with angles either :math:`\pi` (for ones) or 0 (for zeros).

        Args:
            x (ndarray): the input binary data.

        Returns:
            (Circuit): the constructed ``qibo.Circuit``.
        """
        if x.shape[-1] != len(self.qubits):
            raise_error(
                RuntimeError,
                f"Invalid input dimension {x.shape[-1]}, but the allocated qubits are {self.qubits}.",
            )
        circuit = self.circuit
        x = x.ravel()
        for i, q in enumerate(self.qubits):
            circuit.add(gates.RX(q, theta=x[i] * np.pi, trainable=False))
        return circuit

    @property
    def differentiable(self) -> bool:
        return False
