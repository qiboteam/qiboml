import matplotlib.pyplot as plt
import numpy as np
import qibo
from qibo import hamiltonians, optimizers

from qiboml.models.u3 import ReuploadingU3

qibo.set_backend("numpy")

nqubits = 1

model = ReuploadingU3(nqubits=nqubits, nlayers=5, data_dimensionality=(1,))
h = hamiltonians.Z(nqubits=1)

print(model.circuit.draw())

p = np.random.randn(model.nparams)
model.set_parameters(p)


x_data = np.linspace(0.001, 2, 100)
y_data = np.exp(-((x_data - 1) ** 2) / (2 * 0.1**2))

plt.figure(figsize=(5, 5 * 6 / 8))
plt.plot(x_data, y_data)
plt.savefig("data.png")


def predict(model, x):
    model.inject_data(x)
    return h.expectation(model.circuit.execute().state())


def one_loss(model, x, y):
    model.inject_data(x)
    pred = predict(model, x)
    return (pred - y) ** 2


def loss_function(parameters, model, x_set, y_set):
    loss = 0
    model.set_parameters(parameters)
    for x, y in zip(x_set, y_set):
        loss += one_loss(model, x, y)
    print(loss)
    return loss / len(y_set)


result = optimizers.optimize(
    loss=loss_function,
    args=(model, x_data, y_data),
    initial_parameters=p,
    method="BFGS",
)

model.set_parameters(result[1])

predictions = []
for x in x_data:
    predictions.append(predict(model, x))

plt.figure(figsize=(5, 5 * 6 / 8))
plt.plot(x_data, y_data, label="Target", color="blue", alpha=0.7, lw=1.5)
plt.plot(x_data, predictions, label="Predictions", color="red", alpha=0.7, lw=1.5)
plt.legend()
plt.savefig("preds.png")

print(result)
