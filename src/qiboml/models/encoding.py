import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import cached_property
from typing import Optional

import numpy as np
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

    @cached_property
    def _data_to_gate(self):
        """Mapping between each input component and the corresponding gates it contributes to."""
        raise_error(
            NotImplementedError,
            f"_data_to_gate method is not implemented for encoding {self}.",
        )

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


@dataclass
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

    @cached_property
    def _data_to_gate(self):
        """
        Associate each data component with its index in the gates queue.
        In this case, the correspondence it's simply that the i-th component
        of the data is uploaded in the i-th gate of the queue.
        """
        return {f"{i}": [i] for i in range(len(self.qubits))}

    def __call__(self, x: ndarray) -> Circuit:
        """Construct the circuit encoding the ``x`` data in the chosen encoding gate.

        Args:
            x (ndarray): the input real data to encode in rotation angles.

        Returns:
            (Circuit): the constructed ``qibo.Circuit``.
        """
        if x.shape[-1] != len(self.qubits):
            raise_error(
                RuntimeError,
                f"Invalid input dimension {x.shape[-1]} for the allocated qubits {self.qubits}.",
            )
        if x.ndim == 1:
            x = [x]
        circuits = []
        for i in range(len(x)):
            circuit = self.circuit.copy()
            for j, q in enumerate(self.qubits):
                this_gate_params = {"trainable": False}
                [
                    this_gate_params.update({p: x[i][j]})
                    for p in self.gate_encoding_params
                ]
                circuit.add(self.encoding_gate(q=q, **this_gate_params))
            circuits.append(circuit)
        return circuits


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
                f"Invalid input dimension {x.shape[-1]} for the allocated qubits {self.qubits}.",
            )
        if x.ndim == 1:
            x = [x]
        circuits = []
        for i in range(len(x)):
            circuit = self.circuit.copy()
            for j, q in enumerate(self.qubits):
                circuit.add(gates.RX(q, theta=x[i][j] * np.pi, trainable=False))
            circuits.append(circuit)
        return circuit

    @property
    def differentiable(self) -> bool:
        return False
