"""Keras interface to qiboml layers"""

from dataclasses import dataclass
from typing import List, Optional, Union

import keras
import numpy as np
import tensorflow as tf  # pylint: disable=import-error
from qibo import Circuit
from qibo.backends import Backend
from qibo.config import raise_error

from qiboml.interfaces import utils
from qiboml.models.decoding import QuantumDecoding
from qiboml.models.encoding import QuantumEncoding
from qiboml.operations.differentiation import Differentiation, Jax

DEFAULT_DIFFERENTIATION = {
    "qiboml-pytorch": None,
    "qiboml-tensorflow": None,
    "qiboml-jax": None,
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
            by sequentially add the elements of the given list. It is also possible
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

        params = utils.get_params_from_circuit_structure(self.circuit_structure)
        params = keras.ops.cast(params, "float64")  # pylint: disable=no-member

        self.circuit_parameters = self.add_weight(
            shape=params.shape, initializer="zeros", trainable=True
        )
        self.set_weights([params])

        if self.differentiation is None:
            self.differentiation = utils.get_default_differentiation(
                decoding=self.decoding,
                instructions=DEFAULT_DIFFERENTIATION,
            )

    def call(self, x: tf.Tensor) -> tf.Tensor:
        if self.differentiation is None:
            circuit = utils.circuit_from_structure(
                circuit_structure=self.circuit_structure,
                x=x,
            )
            # this 1 * is needed otherwise a TypeError is raised
            circuit.set_parameters(1 * self.circuit_parameters)

            output = self.decoding(circuit)
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
