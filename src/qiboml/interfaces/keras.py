"""Keras interface to qiboml layers"""

# pylint: disable=no-member
# pylint: disable=unexpected-keyword-arg

import string
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
    "qiboml-pytorch": Jax,  # None,
    "qiboml-tensorflow": None,
    "qiboml-jax": Jax,  # None,
    "numpy": Jax,
}

QIBO_2_KERAS_BACKEND = {
    "qiboml (pytorch)": "torch",
    "qiboml (tensorflow)": "tensorflow",
    "qiboml (jax)": "jax",
    "numpy": "numpy",
}


@dataclass(eq=False)
class QuantumModel(keras.Model):  # pylint: disable=no-member

    encoding: QuantumEncoding
    circuit: Circuit
    decoding: QuantumDecoding
    differentiation: Optional[Differentiation] = None

    def __post_init__(self):
        super().__init__()

        # directly building the weigths in the init as they don't depend
        # on the inputs
        params = [
            self.backend.to_numpy(p)
            for param in self.circuit.get_parameters()
            for p in param
        ]
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

    def compute_output_shape(self, input_shape):
        return self.decoding.output_shape

    def call(self, x: tf.Tensor) -> tf.Tensor:
        # this 1 * is needed otherwise a TypeError is raised
        self.circuit.set_parameters(1 * self.circuit_parameters)

        if self.differentiation is None:
            output = self.decoding(self.encoding(x) + self.circuit)
            return output[None, :]
        return self.custom_gradient.evaluate(x, 1 * self.circuit_parameters)

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
    def evaluate(self, x, params):
        # check whether we have to derive wrt inputs
        wrt_inputs = self.encoding.differentiable
        if tf.is_symbolic_tensor(x):  # how to check if tf.tensor is leaf?
            wrt_inputs = wrt_inputs and hasattr(x, "op") and len(x.op.inputs) > 0

        def forward(x, params):
            x = self.backend.cast(
                keras.ops.convert_to_numpy(x), dtype=self.backend.np.float64
            )
            circuit = self.encoding(x) + self.circuit
            circuit.set_parameters(params)
            y = self.decoding(circuit)
            y = self.backend.to_numpy(y)
            return keras.ops.cast(y, dtype=y.dtype)

        y = tf.numpy_function(func=forward, inp=[x, params], Tout=tf.float64)
        if tf.is_symbolic_tensor(y):
            y.set_shape(self.decoding.output_shape)
        else:
            # check output shape of decoding layers, they returned tensor
            # shape should match the output_shape attribute and this should
            # not be necessary!!
            y = keras.ops.reshape(y, self.decoding.output_shape)

        def get_gradients(x, params):
            d_x, *d_params = self.differentiation.evaluate(
                x,
                self.encoding,
                self.circuit,
                self.decoding,
                self.backend,
                *params,
                wrt_inputs=wrt_inputs,
            )
            d_params = self.backend.to_numpy(
                self.backend.cast(d_params, dtype=self.backend.np.float64)
            )
            d_x = self.backend.to_numpy(d_x)
            d_x = keras.ops.cast(d_x, d_x.dtype)
            d_params = keras.ops.cast(d_params, d_params.dtype)
            return d_x, d_params

        # Custom gradient
        def grad(dy):
            d_x, d_params = tf.numpy_function(
                func=get_gradients, inp=[x, params], Tout=[tf.float64, tf.float64]
            )
            if tf.is_symbolic_tensor(d_x):
                d_x.set_shape(dy.shape + x.shape)
                d_params.set_shape(tuple(params.shape) + dy.shape)
            else:
                # double check this
                # the reshape here should not be needed
                d_x = keras.ops.reshape(d_x, dy.shape + x.shape)
                d_params = keras.ops.reshape(d_params, tuple(params.shape) + dy.shape)
            indices = tuple(range(len(dy.shape)))
            lhs = "".join(string.ascii_letters[i] for i in indices)
            rhs = string.ascii_letters[len(indices)] + lhs
            d_params = keras.ops.einsum(f"{lhs},{rhs}", dy, d_params)
            rhs = lhs + "".join(
                string.ascii_letters[i] for i in range(len(indices), len(d_x.shape))
            )
            d_x = keras.ops.einsum(f"{lhs},{rhs}", dy, d_x)
            return d_x, d_params

        return y, grad
