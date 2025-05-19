import inspect
from abc import ABC
from dataclasses import dataclass, field
from functools import cached_property
from typing import Optional, Union

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
        encoding_rule (Optional[Union["torch.nn.module", "keras.layers.Layer"]]): optional
            trainable encoding rule which can be used to preprocess the data with some
            classical model.
    """

    nqubits: int
    qubits: Optional[tuple[int]] = None
    encoding_rule: Optional[Union["torch.nn.module", "keras.layers.Layer"]] = None

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

    @cached_property
    def _data_to_gate(self):
        raise_error(
            NotImplementedError,
            f"_data_to_gate method is not implemented for encoding {self}.",
        )

    def __call__(self, x: ndarray) -> Circuit:
        """Abstract call method."""
        if self.encoding_rule is not None:
            return self.encoding_rule(x)
        return x

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
        # Applying encoding rule if we have one
        x = super().__call__(x)

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

        x = super().__call__(x)

        circuit = self.circuit
        x = x.ravel()
        for i, q in enumerate(self.qubits):
            circuit.add(gates.RX(q, theta=x[i] * np.pi, trainable=False))
        return circuit

    @property
    def differentiable(self) -> bool:
        return False
