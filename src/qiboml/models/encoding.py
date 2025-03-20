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
        # Dictionary which helps to map each data component into a gate in the circuit
        self._data_to_gate = {}

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

        signature = inspect.signature(self.encoding_gate)
        allowed_params = {"theta", "phi", "lam"}
        gate_params = {p for p in signature.parameters.keys()} & allowed_params

        if len(gate_params) != 1:
            raise ValueError(f"{self} currently support only gates with one parameter.")

        # Construct initial 0 values for the gate's parameters
        params = {param: 0.0 for param in gate_params}
        params.update({"trainable": False})

        for q in self.qubits:
            self._circuit.add(self.encoding_gate(q, **params))

    @cached_property
    def _data_to_gate(self):
        """
        Associate each data component with its index in the gates queue.
        In this case, the correspondence it's simply that the i-th component
        of the data is uploaded in the i-th gate of the queue.
        """
        return {f"{i}": [i] for i in range(len(self._circuit.parametrized_gates))}

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
        self._set_phases(x)
        return self.circuit


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
        for qubit, bit in zip(self.qubits, x.ravel()):
            circuit.add(gates.RX(qubit, theta=bit * np.pi, trainable=False))
        return circuit

    @property
    def differentiable(self) -> bool:
        return False
