import os

import keras
import matplotlib.pyplot as plt
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf

tf.get_logger().setLevel("ERROR")


import qibo

import qiboml
import qiboml.models
import qiboml.models.layers
from qiboml.models.ansatze import reuploading_circuit

qibo.set_backend("tensorflow")

x = np.linspace(-1, 1, 500)
y = np.sin(6 * x) ** 2 + np.random.normal(0, 0.1, 500)

nqubits = 3
nlayers = 1

circuit = reuploading_circuit(nqubits, nlayers)

model = keras.Sequential()
model.add(keras.layers.Input(shape=(1,)))
model.add(keras.layers.Dense(2**nqubits, activation="relu"))
model.add(qiboml.models.layers.QuantumLayer(circuit=circuit))
model.add(keras.layers.Dense(32, input_shape=(8,), activation="relu"))
model.add(keras.layers.Dense(1, activation="linear"))
model.compile(optimizer="adam", loss="mse")
model.summary()

model.fit(x, y, batch_size=32, epochs=100)

test_x = np.linspace(-1, 1, 100)
test_y = model.predict(test_x)

plt.figure(figsize=(6, 6 * 6 / 8))
plt.plot(x, y, marker=".", color="royalblue", label="Training")
plt.plot(test_x, test_y, marker=".", color="red", label="Test")
plt.legend()
plt.savefig("data.png")
