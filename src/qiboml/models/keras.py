"""Keras interface to qiboml layers"""

import inspect
from dataclasses import dataclass

import keras
import numpy as np
import tensorflow as tf
from keras.src.backend import compute_output_spec

import qiboml.models.ansatze as ans
import qiboml.models.encoding_decoding as ed


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
