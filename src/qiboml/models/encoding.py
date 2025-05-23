import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import cached_property
from typing import Optional

import numpy as np
from qibo import Circuit, gates
from qibo.config import raise_error

from qiboml import ndarray


@dataclass(eq=False)
class QuantumEncoding(ABC):
    """
    Abstract Encoder class.

    Args:
        nqubits (int): total number of qubits.
        qubits (tuple[int], optional): set of qubits it acts on, by default ``range(nqubits)``.
        density_matrix (bool, optional): whether to build the circuit with ``density_matrix=True``, mostly useful for noisy simulations. ``density_matrix=False`` by default.
    """

    nqubits: int
    qubits: Optional[tuple[int]] = None
    density_matrix: Optional[bool] = False
    _circuit: Circuit = None

    def __post_init__(
        self,
    ):
        """Ancillary post initialization for the dataclass object."""
        self.qubits = (
            tuple(range(self.nqubits)) if self.qubits is None else tuple(self.qubits)
        )
        self._circuit = Circuit(self.nqubits, density_matrix=self.density_matrix)

    @cached_property
    def _data_to_gate(self):
        """
        Mapping between the index of the input and the indices of the gates in the
        produced encoding circuit queue, where the input is encoded to.
        For instance, {0: [0,2], 1: [2]}, represents an encoding where the element
        0 of the inputs enters the gates with indices 0 and 2 of the queue, whereas
        the element 1 of the input affects only the the gate in position 2 of the
        queue.
        By deafult, the map reproduces a simple encoding where the
        i-th component of the data is uploaded in the i-th gate of the queue.
        """
        return {f"{i}": [i] for i in range(len(self.qubits))}

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


@dataclass(eq=False)
class PhaseEncoding(QuantumEncoding):
    encoding_gate: type = field(default_factory=lambda: gates.RY)

    def __post_init__(
        self,
    ):
        """Ancillary post initialization: builds the internal circuit with the rotation gates."""
        super().__post_init__()

        # Retrieving information about the given encoding gate
        signature = inspect.signature(self.encoding_gate)
        allowed_params = {"theta", "phi", "lam"}
        self.gate_encoding_params = {
            p for p in signature.parameters.keys()
        } & allowed_params

        if len(self.gate_encoding_params) != 1:
            raise NotImplementedError(
                f"{self} currently support only gates with one parameter."
            )

    def __call__(self, x: ndarray) -> Circuit:
        """Construct the circuit encoding the ``x`` data in the chosen encoding gate.

        Args:
            x (ndarray): the input real data to encode in rotation angles.

        Returns:
            (Circuit): the constructed ``qibo.Circuit``.
        """
        circuit = self.circuit
        x = x.ravel()
        for i, q in enumerate(self.qubits):
            this_gate_params = {"trainable": False}
            [this_gate_params.update({p: x[i]}) for p in self.gate_encoding_params]
            circuit.add(self.encoding_gate(q=q, **this_gate_params))
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
