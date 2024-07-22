"""Keras interface to qiboml layers"""

from dataclasses import dataclass

import keras
import numpy as np

# from keras.src.backend import compute_output_spec
from qibo.config import raise_error

import qiboml.models.ansatze as ans
import qiboml.models.encoding_decoding as ed
from qiboml.models.abstract import QuantumCircuitLayer

# import tensorflow as tf



"""
def _keras_factory(module):
    for name, layer in inspect.getmembers(module, inspect.isclass):
        if layer.__module__ == module.__name__:

            def __init__(cls, *args, **kwargs):
                nonlocal layer
                name = kwargs.pop("name", None)
                keras.layers.Layer.__init__(cls, name=name)
                layer.__init__(cls, *args, **kwargs)
                if len(cls.circuit.get_parameters()) > 0:
                    cls.add_weight(
                        shape=(len(cls.circuit.get_parameters()),),
                        initializer="zeros",
                    )
                    cls.set_weights(
                        [
                            np.hstack(cls.circuit.get_parameters()),
                        ]
                    )

            def compute_output_shape(cls):
                return (cls.nqubits,)

            @tf.custom_gradient
            def call(cls, x):
                return cls.forward(x), cls.backward

            globals()[name] = dataclass(
                type(
                    name,
                    (keras.layers.Layer, layer),
                    {
                        "__init__": __init__,
                        "call": call,
                        "compute_output_shape": compute_output_shape,
                        "__hash__": keras.layers.Layer.__hash__,
                    },
                )
            )


for module in (ed, ans):
    _keras_factory(module)
"""


@dataclass
class QuantumModel(keras.layers.Layer):  # pylint: disable=no-member

    def __init__(self, layers: list[QuantumCircuitLayer]):
        super().__init__()
        nqubits = layers[0].circuit.nqubits
        self.layers = layers
        for layer in layers[1:]:
            if layer.circuit.nqubits != nqubits:
                raise_error(
                    RuntimeError,
                    f"Layer \n{layer}\n has {layer.circuit.nqubits} qubits, but {nqubits} qubits was expected.",
                )
        if not isinstance(layers[-1], ed.QuantumDecodingLayer):
            raise_error(
                RuntimeError,
                f"The last layer has to be a `QuantumDecodinglayer`, but is {layers[-1]}",
            )

    def call(self, x: "ndarray"):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def compute_output_shape(self):
        return self.layers[-1].output_shape

    @property
    def nqubits(self):
        return self.layers[0].circuit.nqubits

    def __hash__(self):
        return super().__hash__()
