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


@tf.custom_gradient
def custom_operation(
    encoding, circuit, decoding, differentiation, backend, parameters, x
):
    """
    We need to detach the parameters and the datapoint from the
    TensorFlow graph.
    """
    # Datapoint
    x_clone = tf.identity(x)
    x_clone = tf.stop_gradient(x)

    with tf.device("CPU:0"):
        x_clone = tf.identity(x_clone)

    x_clone = x_clone.numpy()
    x_clone = backend.cast(x_clone, dtype=backend.precision)

    # Parameters
    parameters = tf.identity(parameters)
    params = []
    for w in parameters:
        w_clone = tf.identity(w)
        w_clone = tf.stop_gradient(w_clone)

        with tf.device("CPU:0"):
            w_clone = tf.identity(w_clone)

        w_clone_numpy = w_clone.numpy()
        w_clone_backend_compatible = backend.cast(
            w_clone_numpy, dtype=backend.precision
        )
        params.append(w_clone_backend_compatible)

    output = encoding(x_clone) + circuit
    output.set_parameters(params)
    output = decoding(output)
    output = tf.expand_dims(output, axis=0)

    def custom_grad(upstream):
        grad_input, *gradients = (
            tf.Constant(backend.to_numpy(grad).tolist())
            for grad in differentiation.evaluate(
                x_clone, encoding, circuit, decoding, backend, *parameters
            )
        )

        return upstream * grad_input

    return output, custom_grad


@dataclass(eq=False)
class QuantumModel(keras.Model):  # pylint: disable=no-member

    encoding: QuantumEncoding
    circuit: Circuit
    decoding: QuantumDecoding
    differentiation: str = "auto"

    def __post_init__(self):
        super().__init__()

        # Trainable parameters
        # Prendo i parametri da self.circuit perché mi interessa la shape per
        # generere in modo gaussiano self.circuit_parameters
        params = [p for param in self.circuit.get_parameters() for p in param]
        params = tf.Variable(self.backend.to_numpy(params))

        self.circuit_parameters = self.add_weight(
            shape=params.shape, initializer="random_normal", trainable=True
        )

    def call(self, x: tf.Tensor) -> tf.Tensor:
        if (
            self.backend.name != "tensorflow"
            or self.differentiation is not None
            or not self.decoding.analytic
        ):

            """devo chiamare una funzione che mi ritorna la prediction voluta e
            la funzione custom grad"""
            return custom_operation(
                self.encoding,
                self.circuit,
                self.decoding,
                self.differentiation,
                self.circuit_parameters,
                x,
            )

        else:

            weights = tf.identity(self.circuit_parameters)
            self.circuit.set_parameters(weights)

            y = self.encoding(x) + self.circuit
            output = self.decoding(y)
            output_expanded = tf.expand_dims(output, axis=0)

        return output_expanded

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
