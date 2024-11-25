import qibo
from qibo import gates, Circuit
import numpy as np
import tensorflow as tf


def random_subset(nqubits, k):
    return np.random.choice(range(nqubits), size=(k,), replace=False).tolist()


nqubits = 3
dim = 2
backend = "tensorflow"

c = Circuit(nqubits)
c.add(gates.X(0))
c.add(gates.X(1))
c.add(gates.Z(1))
c.add(gates.CNOT(0, 1))
c.add(gates.RX(0, theta=0.4))

random_choice = random_subset(nqubits, dim)
print(f"Scelta random {random_choice}")
result = c().probabilities()
print(result)


tensor = tf.random.uniform((2, nqubits), minval=0, maxval=2, dtype=tf.int32)
print(f"Tensore: {tensor}")
