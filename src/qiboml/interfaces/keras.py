"""Keras interface to qiboml layers"""

from dataclasses import dataclass
from typing import Optional

import keras
import numpy as np
import tensorflow as tf  # pylint: disable=import-error
from qibo import Circuit
from qibo.backends import Backend
from qibo.config import raise_error

from qiboml.models.decoding import QuantumDecoding
from qiboml.models.encoding import QuantumEncoding
from qiboml.operations.differentiation import PSR, Differentiation, Jax

DEFAULT_DIFFERENTIATION = {
    "qiboml-pytorch": None,
    "qiboml-tensorflow": None,
    "qiboml-jax": None,
    "numpy": Jax,
}


@dataclass(eq=False)
class QuantumModel(keras.Model):  # pylint: disable=no-member

    encoding: QuantumEncoding
    circuit: Circuit
    decoding: QuantumDecoding
    differentiation: Optional[Differentiation] = None

    def __post_init__(self):
        super().__init__()

        params = [p for param in self.circuit.get_parameters() for p in param]
        params = keras.ops.cast(params, "float64")  # pylint: disable=no-member

        self.circuit_parameters = self.add_weight(
            shape=params.shape, initializer="zeros", trainable=True
        )
        self.set_weights([params])

        backend_string = (
            f"{self.decoding.backend.name}-{self.decoding.backend.platform}"
            if self.decoding.backend.platform is not None
            else self.decoding.backend.name
        )
        if self.differentiation is None:
            if not self.decoding.analytic:
                self.differentiation = PSR()
            else:
                if backend_string in DEFAULT_DIFFERENTIATION.keys():
                    diff = DEFAULT_DIFFERENTIATION[backend_string]
                    self.differentiation = diff() if diff is not None else None
                else:
                    self.differentiation = PSR()

    def call(self, x: tf.Tensor) -> tf.Tensor:
        if self.differentiation is None:
            # this 1 * is needed otherwise a TypeError is raised
            self.circuit.set_parameters(1 * self.circuit_parameters)

            output = self.decoding(self.encoding(x) + self.circuit)
            return output[None, :]

        raise NotImplementedError
        # return custom_operation(
        #    self.encoding,
        #    self.circuit,
        #    self.decoding,
        #    self.differentiation,
        #    self.circuit_parameters,
        #    x,
        # )

    @property
    def output_shape(
        self,
    ):
        return self.decoding.output_shape

    @property
    def nqubits(
        self,
    ) -> int:
        return self.encoding.nqubits

    @property
    def backend(
        self,
    ) -> Backend:
        return self.decoding.backend
