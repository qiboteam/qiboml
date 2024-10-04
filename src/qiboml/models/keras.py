"""Keras interface to qiboml layers"""

from dataclasses import dataclass

import keras
import numpy as np
import tensorflow as tf  # pylint: disable=import-error
from qibo import Circuit
from qibo.backends import Backend
from qibo.config import raise_error

# import qiboml.models.encoding_decoding as ed
# from qiboml.models.abstract import QuantumCircuitLayer
from qiboml.models.decoding import QuantumDecoding
from qiboml.models.encoding import QuantumEncoding
from qiboml.operations import differentiation as Diff

BACKEND_2_DIFFERENTIATION = {
    "pytorch": "PSR",
    "tensorflow": None,
    "jax": "PSR",
}

"""
@dataclass
class QuantumModel(keras.Layer):  # pylint: disable=no-member

    layers: list[QuantumCircuitLayer]

    def __post_init__(self):
        super().__init__()
        for layer in self.layers[1:]:
            if layer.circuit.nqubits != self.nqubits:
                raise_error(
                    RuntimeError,
                    f"Layer \n{layer}\n has {layer.circuit.nqubits} qubits, but {self.nqubits} qubits was expected.",
                )
            if layer.backend.name != self.backend.name:
                raise_error(
                    RuntimeError,
                    f"Layer \n{layer}\n is using {layer.backend} backend, but {self.backend} backend was expected.",
                )
        if not isinstance(self.layers[-1], ed.QuantumDecodingLayer):
            raise_error(
                RuntimeError,
                f"The last layer has to be a `QuantumDecodinglayer`, but is {self.layers[-1]}",
            )

    def call(self, x: tf.Tensor) -> tf.Tensor:
        if self.backend.name != "tensorflow":
            if self.backend.name == "pytorch":
                self.backend.requires_grad(False)
            x = self.backend.cast(np.array(x))
        for layer in self.layers:
            x = layer.forward(x)
        if self.backend.name != "tensorflow":
            x = tf.convert_to_tensor(np.array(x))
        return x

    def compute_output_shape(self):
        return self.output_shape

    @property
    def output_shape(self):
        return self.layers[-1].output_shape

    @property
    def nqubits(self) -> int:
        return self.layers[0].circuit.nqubits

    @property
    def backend(self) -> Backend:
        return self.layers[0].backend

    def __hash__(self):
        return super().__hash__()
"""


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
