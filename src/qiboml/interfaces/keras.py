"""Keras interface to qiboml layers"""

# pylint: disable=no-member
# pylint: disable=unexpected-keyword-arg

import string
from dataclasses import dataclass
from typing import List, Optional, Union

import keras
import numpy as np
import tensorflow as tf  # pylint: disable=import-error
from qibo import Circuit
from qibo.backends import Backend

from qiboml.interfaces import utils
from qiboml.models.decoding import QuantumDecoding
from qiboml.models.encoding import QuantumEncoding
from qiboml.operations.differentiation import Differentiation, Jax

DEFAULT_DIFFERENTIATION = {
    "qiboml-pytorch": Jax,
    "qiboml-tensorflow": None,
    "qiboml-jax": Jax,
    "numpy": Jax,
}


@dataclass(eq=False)
class QuantumModel(keras.Model):  # pylint: disable=no-member
    """
    The Keras interface to qiboml models.

    Args:
        circuit_structure (Union[List[QuantumEncoding, Circuit], Circuit]):
            a list of Qibo circuits and Qiboml encoding layers, which defines
            the complete structure of the model. The whole circuit will be mounted
            by sequentially stacking the elements of the given list. It is also possible
            to pass a single circuit, in the case a sequential structure is not needed.
        decoding (QuantumDecoding): the decoding layer.
        differentiation (Differentiation, optional): the differentiation engine,
            if not provided a default one will be picked following what described in the :ref:`docs <_differentiation_engine>`.
    """

    circuit_structure: Union[Circuit, List[Union[Circuit, QuantumEncoding]]]
    decoding: QuantumDecoding
    differentiation: Optional[Differentiation] = None

    def __post_init__(self):
        super().__init__()

        if isinstance(self.circuit_structure, Circuit):
            self.circuit_structure = [self.circuit_structure]
        utils._uniform_circuit_structure_density_matrix(self.circuit_structure)

        params = utils.get_params_from_circuit_structure(self.circuit_structure)
        params = keras.ops.cast(
            self.backend.to_numpy(params), "float64"
        )  # pylint: disable=no-member

        self.circuit_parameters = self.add_weight(
            shape=params.shape, initializer="zeros", trainable=True
        )
        self.set_weights([params])

        if self.differentiation is None:
            self.differentiation = utils.get_default_differentiation(
                decoding=self.decoding,
                instructions=DEFAULT_DIFFERENTIATION,
            )

        if self.differentiation is not None:
            self.custom_gradient = QuantumModelCustomGradient(
                self.circuit_structure,
                self.decoding,
                self.backend,
                self.differentiation,
            )

    def compute_output_shape(self, input_shape):
        return self.decoding.output_shape

    def call(self, x: Optional[tf.Tensor] = None) -> tf.Tensor:
        circuit = utils.circuit_from_structure(
            x=x, circuit_structure=self.circuit_structure
        )

        if self.differentiation is None:
            # This 1 * is needed otherwise a TypeError is raised
            circuit.set_parameters(1 * self.circuit_parameters)
            output = self.decoding(circuit)
            return output[None, :]
        if x is None:
            x = tf.constant([])
        return self.custom_gradient.evaluate(x, 1 * self.circuit_parameters)

    def draw(self, plt_drawing=True, **plt_kwargs):
        """
        Draw the full circuit structure.

        Args:
            plt_drawing (bool): if True, the `qibo.ui.plot_circuit` function is used.
                If False, the default `circuit.draw` method is used.
            plt_kwargs (dict): extra arguments which can be set to customize the
                `qibo.ui.plot_circuit` function.
        """

        fig = utils.draw_circuit(
            circuit_structure=self.circuit_structure,
            backend=self.decoding.backend,
            plt_drawing=plt_drawing,
            **plt_kwargs,
        )

        return fig

    @property
    def output_shape(
        self,
    ):
        return self.decoding.output_shape

    @property
    def nqubits(
        self,
    ) -> int:
        return self.decoding.nqubits

    @property
    def backend(
        self,
    ) -> Backend:
        return self.decoding.backend


@dataclass
class QuantumModelCustomGradient:

    circuit_structure: Union[Circuit, List[Union[Circuit, QuantumEncoding]]]
    decoding: QuantumDecoding
    backend: Backend
    differentiation: Differentiation
    wrt_inputs: bool = False

    @tf.custom_gradient
    def evaluate(self, x, params):
        x_is_not_None = x.shape[0] != 0
        # check whether we have to derive wrt inputs
        if x_is_not_None and tf.is_symbolic_tensor(
            x
        ):  # how to check if tf.tensor is leaf?

            differentiable_encodings = True
            for circ in self.circuit_structure:
                if isinstance(circ, QuantumEncoding):
                    if not circ.differentiable:
                        differentiable_encodings = False

            self.wrt_inputs = (
                differentiable_encodings and hasattr(x, "op") and len(x.op.inputs) > 0
            )

        def forward(x, params):
            if x_is_not_None:
                x = self.backend.cast(
                    keras.ops.convert_to_numpy(x), dtype=self.backend.np.float64
                )
            circuit = utils.circuit_from_structure(
                circuit_structure=self.circuit_structure,
                x=x,
            )
            circuit.set_parameters(params)
            y = self.decoding(circuit)
            y = self.backend.to_numpy(y)
            return keras.ops.cast(y, dtype=y.dtype)

        y = tf.numpy_function(func=forward, inp=[x, params], Tout=tf.float64)
        # check output shape of decoding layers, their returned tensor
        # shape should match the output_shape attribute and this should
        # not be necessary (only useful for symbolic execution)!!
        y = keras.ops.reshape(y, self.decoding.output_shape)

        def get_gradients(x, params):
            if x_is_not_None:
                x = self.backend.cast(x, dtype=self.backend.np.float64)
            d_x, *d_params = self.differentiation.evaluate(
                x,
                self.circuit_structure,
                self.decoding,
                self.backend,
                *params,
                wrt_inputs=self.wrt_inputs,
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
            if x_is_not_None:
                # double check this
                # the reshape here should be needed for symbolic execution only
                d_x = keras.ops.reshape(d_x, dy.shape + x.shape)
            d_params = keras.ops.reshape(d_params, tuple(params.shape) + dy.shape)
            indices = tuple(range(len(dy.shape)))
            lhs = "".join(string.ascii_letters[i] for i in indices)
            rhs = string.ascii_letters[len(indices)] + lhs
            d_params = keras.ops.einsum(f"{lhs},{rhs}", dy, d_params)
            rhs = lhs + "".join(
                string.ascii_letters[i] for i in range(len(indices), len(d_x.shape))
            )
            d_x = keras.ops.einsum(f"{lhs},{rhs}", dy, d_x) if x_is_not_None else None
            return d_x, d_params

        return y, grad
