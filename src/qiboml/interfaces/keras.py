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
        x = quantum_model_gradient(
            x,
            self.encoding,
            self.circuit,
            self.decoding,
            self.backend,
            self.differentiation,
            *self.get_weights()[0],
        )
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


@tf.custom_gradient
def quantum_model_gradient(
    x, encoding, circuit, decoding, backend, differentiation, *params
):
    # check whether we have to derive wrt inputs
    wrt_inputs = x.trainable and encoding.differentiable

    x = backend.cast(keras.ops.convert_to_numpy(x), dtype=backend.np.float64)
    x = encoding(x) + circuit
    x.set_parameters(params)
    y = decoding(x)
    y = keras.ops.cast(y, dtype=y.dtype)

    # Custom gradient
    def grad(dy):
        d_x, d_params = differentiation.evaluate(
            x,
            encoding,
            circuit,
            decoding,
            backend,
            *params,
            wrt_inputs=wrt_inputs,
        )
        # breakpoint()
        return dy * d_x, None, None, None, None, None, dy * d_params

    return y, grad
