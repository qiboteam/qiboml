from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import qibo
from qibo import hamiltonians, optimizers

from qiboml.models.reuploading.fourier import FourierReuploading
from qiboml.models.reuploading.u3 import ReuploadingU3

qibo.set_backend("numpy")

# ---------------------- UTILS --------------------------


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
    return loss / len(y_set)


def mse(labels, predictions):
    """Compute MSE given predictions and labels."""
    return np.sum((labels - predictions) ** 2) / len(labels)


x_data = np.linspace(0.001, 2, 50)
y_data = np.exp(-((x_data - 1) ** 2) / (2 * 0.4**2))

plt.figure(figsize=(5, 5 * 6 / 8))
plt.plot(x_data, y_data)
plt.savefig("data.png")

# ------------------------ ITERATIVE TRAINING ---------------------

nqubits = 1
nlayers = 3
nruns = 10

mse_list = []

for run in range(nruns):
    print(f"Running training {run+1}/{nruns}")
    # model = ReuploadingU3(nqubits=nqubits, nlayers=nlayers, data_dimensionality=(1,))
    model = FourierReuploading(
        nqubits=nqubits, nlayers=nlayers, data_dimensionality=(1,)
    )

    print(model.parameters)

    h = hamiltonians.Z(nqubits=1)

    print(model.circuit.draw())

    p = np.random.randn(model.nparams)
    initial_p = deepcopy(p)
    model.set_parameters(p)

    result = optimizers.optimize(
        loss=loss_function,
        args=(model, x_data, y_data),
        initial_parameters=p,
        method="BFGS",
    )

    predictions = []
    for x in x_data:
        predictions.append(predict(model, x))

    plt.figure(figsize=(5, 5 * 6 / 8))
    plt.plot(predictions, label="predictions")
    plt.plot(y_data, label="labels")
    plt.savefig(f"u3_run_{run}.png")

    mse_list.append(mse(y_data, predictions))

mean_mse = np.mean(mse_list)
std_mse = np.std(mse_list)

print(f"MSE estimation: {mean_mse} +- {std_mse}")
