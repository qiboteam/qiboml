"""Keras interface to qiboml layers"""

import string
from dataclasses import dataclass
from typing import Optional

import keras
import numpy as np
import tensorflow as tf  # pylint: disable=import-error
from qibo import Circuit
from qibo.backends import Backend

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
        if self.differentiation is not None:
            self.custom_gradient = QuantumModelCustomGradient(
                self.encoding,
                self.circuit,
                self.decoding,
                self.backend,
                self.differentiation,
            )

    def call(self, x: tf.Tensor) -> tf.Tensor:
        if self.differentiation is None:
            # this 1 * is needed otherwise a TypeError is raised
            self.circuit.set_parameters(1 * self.circuit_parameters)

            output = self.decoding(self.encoding(x) + self.circuit)
            return output[None, :]
        """
        x = quantum_model_gradient(
            x,
            self.encoding,
            self.circuit,
            self.decoding,
            self.backend,
            self.differentiation,
            *self.get_weights()[0],
        )
        """
        x = self.custom_gradient.evaluate(x, *self.get_weights()[0])
        return x

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


@dataclass
class QuantumModelCustomGradient:

    encoding: QuantumEncoding
    circuit: Circuit
    decoding: QuantumDecoding
    backend: Backend
    differentiation: Differentiation

    @tf.custom_gradient
    def evaluate(self, x, *params):
        # check whether we have to derive wrt inputs
        wrt_inputs = x.trainable and self.encoding.differentiable

        x = self.backend.cast(
            keras.ops.convert_to_numpy(x), dtype=self.backend.np.float64
        )
        circuit = self.encoding(x) + self.circuit
        circuit.set_parameters(params)
        y = self.decoding(circuit)
        y = keras.ops.cast(y, dtype=y.dtype)

        # Custom gradient
        def grad(dy):
            d_x, *d_params = self.differentiation.evaluate(
                x,
                self.encoding,
                self.circuit,
                self.decoding,
                self.backend,
                *params,
                wrt_inputs=wrt_inputs,
            )
            d_params = self.backend.cast(d_params, dtype=self.backend.np.float64)
            d_x = keras.ops.cast(d_x, d_x.dtype)
            d_params = keras.ops.cast(d_params, d_params.dtype)
            left_indices = tuple(range(len(d_params.shape)))
            right_indices = left_indices[::-1][: len(d_params.shape) - 2] + (
                len(left_indices),
            )
            left_indices = "".join(string.ascii_letters[i] for i in left_indices)
            right_indices = "".join(string.ascii_letters[i] for i in right_indices)
            d_params = keras.ops.einsum(
                f"{left_indices},{right_indices}", d_params, dy.T
            )
            return dy @ d_x, *d_params

        return y, grad
