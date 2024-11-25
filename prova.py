
import tensorflow as tf
import numpy as np
from qibo import Circuit, gates


@tf.custom_gradient
def custom_operation():
    output = 

    def grad_fn()


    return output, grad_fn



class MyLayer(tf.keras.layers.Layer):

    def __init__(self):
        super(MyLayer, self).__init__():
        self.circuit = self.circuit()

        self.weights = self.add_weights(name='w', shape=(4,), initializer="random_normal")


    def circuit(self):
        c = Circuit(2)
        c.add(gates.X(0))
        c.add(gates.RX(1, theta=0.5))

    def call(self, x):
        self.circuit()

    def 
