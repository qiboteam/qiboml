"""Keras interface to qiboml layers"""

from dataclasses import dataclass

import numpy as np
import tensorflow as tf  # pylint: disable=import-error
from keras.layers import Layer  # pylint: disable=import-error, no-name-in-module
from qibo.backends import Backend
from qibo.config import raise_error

import qiboml.models.encoding_decoding as ed
from qiboml.models.abstract import QuantumCircuitLayer


@dataclass
class QuantumModel(Layer):  # pylint: disable=no-member

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
