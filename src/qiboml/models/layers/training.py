import keras
import qibo
import tensorflow as tf


class QuantumLayer(keras.layers.Layer):
    def __init__(self, circuit: qibo.Circuit, **kwargs):
        super().__init__(**kwargs)
        self.circuit = circuit
        nparams = len(circuit.get_parameters())
        self.qparams = self.add_weight(
            shape=(nparams,),
            initializer="random_normal",
            trainable=True,
            dtype="float64",
        )

    def call(self, inputs):
        # Check input shape
        if inputs.shape[1] != 2**self.circuit.nqubits:
            raise ValueError(
                f"Input of a QuantumLayer has to be a vector of length 2**{self.circuit.nqubits}"
            )
        # Update circuit parameters with trainable weights
        self.circuit.set_parameters(self.qparams.value)
        # Execute circuit
        state = self.circuit(initial_state=inputs).state()
        # Ensure return type is compatible with TensorFlow
        return tf.abs(state)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 2**self.circuit.nqubits)
