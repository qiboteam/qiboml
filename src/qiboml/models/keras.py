"""Keras interface to qiboml layers"""

from dataclasses import dataclass

import keras
import numpy as np
import tensorflow as tf  # pylint: disable=import-error
from qibo import Circuit
from qibo.backends import Backend
from qibo.config import raise_error

from qiboml.models.decoding import QuantumDecoding
from qiboml.models.encoding import QuantumEncoding
from qiboml.operations import differentiation as Diff

BACKEND_2_DIFFERENTIATION = {
    "pytorch": "PSR",
    "tensorflow": None,
    "jax": "PSR",
}


@dataclass(eq=False)
class QuantumModel(keras.Model):  # pylint: disable=no-member

    encoding: QuantumEncoding
    circuit: Circuit
    decoding: QuantumDecoding
    differentiation: str = "auto"

    def __post_init__(self):
        super().__init__()

        params = [p for param in self.circuit.get_parameters() for p in param]
        params = tf.Variable(self.backend.to_numpy(params))
        self.circuit_parameters = self.add_weight(shape=params.shape, trainable=True)
        self.set_weights([params])

    def call(self, x: tf.Tensor) -> tf.Tensor:
        if self.backend.name != "tensorflow":
            pass
        # @tf.custom_gradient
        # def custom_call(x: tf.Tensor):
        #    x = self.backend.cast(np.array(x))

        else:
            self.circuit.set_parameters(self.get_weights()[0])
            # self.circuit.set_parameters(self.circuit_parameters)
            x = self.encoding(x) + self.circuit
            x = self.decoding(x)

        return x

    def compute_output_shape(
        self,
    ):
        return self.output_shape

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
