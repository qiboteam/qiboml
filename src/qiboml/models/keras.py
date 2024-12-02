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
    "tensorflow": "PSR",
    "jax": "PSR",
}


@dataclass()
class CustomGrad:
    _encoding: QuantumEncoding
    _circuit: Circuit
    _decoding: QuantumDecoding
    _differentiation: None
    _backend: Backend
    _parameters: None

    def __post_init__(self):
        self._parameters = tf.unstack(tf.identity(self._parameters))

    def __call__(self, x):
        @tf.custom_gradient
        def custom_gradient(x):

            breakpoint()

            def forward(x):
                return self._decoding(self._encoding(x) + self._circuit)

            def grad_fn(upstream):
                breakpoint()
                # x = tf.identity(x)
                grad_input, *gradients = self._differentiation.evaluate(
                    x,
                    self._encoding,
                    self._circuit,
                    self._decoding,
                    self._backend,
                    *self._parameters,
                )

                # upstream ha la shape dello output di custom_gradient che è forward(x) shape=(4,)
                # grad_fn deve avere la shape dell'input di custom_gradient: shape=(1,2)

                breakpoint()
                print(f"Upstream {upstream}")
                return (upstream @ grad_input, *gradients)

            return forward(x), grad_fn

        return custom_gradient(x)


@dataclass(eq=False)
class QuantumModel(keras.Model):  # pylint: disable=no-member

    encoding: QuantumEncoding
    circuit: Circuit
    decoding: QuantumDecoding
    differentiation: None

    def __post_init__(self):
        super().__init__()

        params = [p for param in self.circuit.get_parameters() for p in param]
        params = tf.Variable(self.backend.to_numpy(params))

        self.circuit_parameters = self.add_weight(
            shape=params.shape, initializer="random_normal", trainable=True
        )

    def call(self, x: tf.Tensor) -> tf.Tensor:
        if self.backend.platform != "tensorflow" or self.differentiation is not None:

            custom = CustomGrad(
                self.encoding,
                self.circuit,
                self.decoding,
                self.differentiation,
                self.backend,
                self.circuit_parameters,
            )
            a = custom(x)
            breakpoint()
            return a

        else:

            weights = tf.identity(self.circuit_parameters)
            self.circuit.set_parameters(weights)

            output = self.decoding(self.encoding(x) + self.circuit)
            output = tf.expand_dims(output, axis=0)
            return output

    def compute_output_shape(
        self,
    ):
        return self.output_shape

    def draw(
        self,
    ):
        breakpoint()
        print("ciao")

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
