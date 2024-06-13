import inspect
from dataclasses import dataclass

import keras
import tensorflow as tf

import qiboml.models.ansatze as ans
import qiboml.models.encoding_decoding as ed


def _keras_factory(module):
    for name, layer in inspect.getmembers(module, inspect.isclass):
        if layer.__module__ == module.__name__:

            def __init__(cls, *args, **kwargs):
                nonlocal layer
                keras.layers.Layer.__init__(cls)
                layer.__init__(cls, *args, **kwargs)
                if len(cls.circuit.get_parameters()) > 0:
                    print("> How do you register parameters with Keras?")

            @tf.custom_gradient
            def call(cls, x):
                nonlocal layer
                return layer.forward(cls, x), layer.backward

            globals()[name] = dataclass(
                type(
                    name,
                    (keras.layers.Layer, layer),
                    {
                        "__init__": __init__,
                        "call": call,
                    },
                )
            )


for module in (ed, ans):
    _keras_factory(module)
