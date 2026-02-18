from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property, partial
from typing import Callable, Optional

import numpy as np
from numpy.typing import ArrayLike
from qibo import Circuit
from qibo.backends import construct_backend

from qiboml.models.decoding import QuantumDecoding


@dataclass
class Differentiation(ABC):
    """
    Abstract differentiator object.
    """

    circuit: Optional[Circuit] = None
    decoding: Optional[QuantumDecoding] = None
    _is_built: bool = False

    def __post_init__(self):
        if self.circuit is not None and self.decoding is not None:
            self.build(self.circuit, self.decoding)

    def build(self, circuit: Circuit, decoding: QuantumDecoding):
        """Attach model internals and prepare compiled artifacts."""
        if self._is_built:  # pragma: no cover
            return
        self.circuit = circuit
        self.decoding = decoding
        self._on_build()
        self._is_built = True

    def _on_build(self) -> None:
        pass

    @abstractmethod
    def evaluate(
        self, parameters: ArrayLike, wrt_inputs: bool = False
    ):  # pragma: no cover
        """
        Evaluate the gradient of the quantum circuit w.r.t its parameters, i.e. its rotation angles.
        """
        pass

    @property
    def backend(self):
        assert self.decoding is not None, "Differentiator not built yet."
        return self.decoding.backend

    @cached_property
    def non_trainable_gates(self):
        assert self.circuit is not None, "Differentiator not built yet."
        return [g for g in self.circuit.parametrized_gates if not g.trainable]

    def nparams(self, wrt_inputs):
        assert self.circuit is not None, "Differentiator not built yet."
        return len(self.circuit.get_parameters(include_not_trainable=wrt_inputs))
