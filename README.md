# Qiboml

👋 Welcome to Qiboml, the quantum machine learning package of the Qibo ecosystem!

---

🎯 Our goal is to integrate Qibo within the most commonly used machine learning frameworks,
allowing the definition and usage of quantum or hybrid classical-quantum models
keeping the same high-level language proposed by the most used libraries (Pytorch, Tensorflow).

![qiboml](https://github.com/user-attachments/assets/c88fd9a7-2511-4672-a911-5d8937dc5d08)

---

### Documentation

[![docs](https://github.com/qiboteam/qibo/actions/workflows/publish.yml/badge.svg)](https://qibo.science/qibo/stable/)

📖 The Qiboml documentation can be found in our [organization webpage](https://qibo.science/qibo/stable/).

---

### Minimum working example

Let's suppose you really like Pytorch interface and you would like to train a Variational Quantum Eigensolver
adopting a full Pytorch pipeline. Here we go! We can start by constructing a target Hamiltonian with Qibo!
For example, we can choose a one-dimensional Heisenberg Hamiltonian with gap $\Delta=0.5$.

```python
from qibo import set_backend
from qibo.hamiltonians import XXZ

# We select the pytorch backend provided by Qiboml
set_backend(backend="qiboml", platform="pytorch")

# Fix the size of the problem
nqubits = 5

hamiltonian = XXZ(nqubits=nqubits, delta=0.5)
```

Then, we can construct our model building a parametrized quantum circuit. Any Qibo
circuit is supported, but we use here a pre-computed ansatz from Qiboml directly.

```python
from qiboml.models.ansatze import HardwareEfficient

# It is a layered ansatz, namely a circuit where we repeat some sequences of gates
circuit = HardwareEfficient(nqubits=nqubits, nlayers=3)
```

Now we need a decoding strategy, to process the state prepared by `circuit`.
Since our goal is to train a VQE, our decoding strategy will be computing the
expectation value of the target Hamiltonian over the state prepared by `circuit`, and
we will train the parameters of the model to minimize it.

```python
from qiboml.models.decoding import Expectation

decoding = Expectation(
    nqubits=nqubits,
    observable=hamiltonian,
)
```

Now that all the pieces are prepared, we can build the quantum model.

```python
# We take the one implemented to be integrated within a Pytorch interface
# but in an equivalent way, a similar model can be imported for Tensorflow
from qiboml.interfaces.pytorch import QuantumModel

qmodel = QuantumModel(
    circuit_structure=training,
    decoding=decoding,
)
```

The circuit structure for our VQE is very simple, but more in general it can be
any sequence of Qibo circuits and Qiboml encoders.

We are finally ready to train it! From now on, everything is pure Pytorch 🤙

```python
import pytorch.optim as optim

# Select the optimizer
optimizer = optim.Adam(qmodel.parameters(), lr=0.05)

# And train!
for iteration in range(300):
    optimizer.zero_grad()
    cost = qmodel()
    cost.backward()
    optimizer.step()
```
