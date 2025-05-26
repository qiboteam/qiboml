# Qiboml

👋 Welcome to Qiboml, the quantum machine learning package of the Qibo ecosystem!

---

🎯 Our goal is to integrate Qibo within the most commonly used machine learning frameworks,
allowing the definition and usage of quantum or hybrid classical-quantum models
keeping the same high-level language proposed by the most used libraries (Pytorch, Tensorflow).

![qiboml](https://github.com/user-attachments/assets/c88fd9a7-2511-4672-a911-5d8937dc5d08)


### Documentation

[![docs](https://github.com/qiboteam/qiboml/actions/workflows/publish.yml/badge.svg)](https://qibo.science/qiboml/stable/)

📖 The Qiboml documentation can be found in our [organization webpage](https://qibo.science/qiboml/stable/).


### Minimum working example

You can quickly build a QML model using one of the currently supported interfaces. For instance,
to train a VQE to find the ground state of an Hamiltonian $H=\sum_i Z_i$:

```python
from qiboml.models.ansatze import HardwareEfficient
from qiboml.models.decoding import Expectation

nqubits = 2
circuit = HardwareEfficient(nqubits)
# By default Expectation sets Z_0 + Z_1 + ... + Z_n as observable,
# any Hamiltonian can be used though
decoding = Expectation(nqubits)

# using pytorch
import torch
import qiboml.interfaces.pytorch as pt

pt_model = pt.QuantumModel(circuit_structure=[circuit,], decoding=decoding)
optimizer = torch.optim.Adam(pt_model.parameters(), lr=0.05)
for iteration in range(100):
    optimizer.zero_grad()
    cost = pt_model()
    cost.backward()
    optimizer.step()

# using keras
import keras
import tensorflow as tf
import qiboml.interfaces.keras as ks
tf.keras.backend.set_floatx('float64') # set the dtype to float64, which is qibo's default

ks_model = ks.QuantumModel(circuit_structure=[circuit,], decoding=decoding)
optimizer = keras.optimizers.Adam(lr=0.05)
for iteration in range(100):
    with tf.GradientTape() as tape:
        cost = ks_model(x)
    gradients = tape.gradient(
        cost, ks_model.trainable_variables
    )
    optimizer.apply_gradients(zip(gradients, ks_model.trainable_variables))
```


### Citation policy

If you use the package please refer to [the documentation](https://qibo.science/qibo/stable/appendix/citing-qibo.html#publications) for citation instructions.

### Contacts

To get in touch with the community and the developers, consider joining the Qibo workspace on Matrix:

[![Matrix](https://img.shields.io/matrix/qibo%3Amatrix.org?logo=matrix)](https://matrix.to/#/#qiboml:matrix.org)

If you have a question about the project, please contact us with [📫](mailto:qiboteam@qibo.science).
