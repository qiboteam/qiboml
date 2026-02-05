import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.typing import ArrayLike
from qibo import Circuit, gates
from qibo.config import raise_error


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

    @abstractmethod
    def __call__(self, x: ArrayLike) -> Circuit:
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

    def __call__(self, x: ArrayLike) -> Circuit:
        """Construct the circuit encoding the ``x`` data in the chosen encoding gate.

        Args:
            x (ArrayLike): the input real data to encode in rotation angles.

        Returns:
            :class:`qibo.models.circuit.Circuit`: The constructed circuit.
        """
        circuit = self.circuit
        if len(x.shape) > 1:
            x = x[0]
        for i, q in enumerate(self.qubits):
            this_gate_params = {"trainable": False}
            [this_gate_params.update({p: x[i]}) for p in self.gate_encoding_params]
            circuit.add(self.encoding_gate(q=q, **this_gate_params))
        return circuit


class BinaryEncoding(QuantumEncoding):

    def __call__(self, x: ArrayLike) -> Circuit:
        r"""Construct the circuit encoding the ``x`` binary data in some ``RX`` rotation gates
        with angles either :math:`\pi` (for ones) or 0 (for zeros).

        Args:
            x (ArrayLike): the input binary data.

        Returns:
            :class:`qibo.models.circuit.Circuit`: The constructed circuit.
        """
        if x.shape[-1] != len(self.qubits):
            raise_error(
                RuntimeError,
                f"Invalid input dimension {x.shape[-1]}, but the allocated qubits are {self.qubits}.",
            )
        circuit = self.circuit
        if len(x.shape) > 1:
            x = x[0]
        for i, q in enumerate(self.qubits):
            circuit.add(gates.RX(q, theta=x[i] * np.pi, trainable=False))
        return circuit

    @property
    def differentiable(self) -> bool:
        return False
