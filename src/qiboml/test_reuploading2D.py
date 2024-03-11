from copy import deepcopy

import matplotlib.pyplot as plt

import numpy as np
import qibo
from qibo import hamiltonians, optimizers

from qiboml.models.reuploading.fourier import FourierReuploading
from qiboml.models.reuploading.u3 import ReuploadingU3

import datetime

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

ndata=10
nvariables=2
x1_range = (0.001, 2)
x2_range = (0.001, 2)
x1_data = np.linspace(x1_range[0], x1_range[1], ndata)
x2_data = np.linspace(x2_range[0], x2_range[1], ndata)

x1_mesh,x2_mesh=np.meshgrid(x1_data, x2_data)
y_mesh = np.exp(-((x1_mesh - 1) ** 2+(x2_mesh - 1) ** 2) / (2 * 0.4**2))

x_data = np.stack((x1_mesh, x2_mesh), axis=-1).reshape(-1, nvariables)
y_data=y_mesh.flatten()

plt.figure(figsize=(5, 5 * 6 / 8))
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x1_mesh, x2_mesh, y_mesh, cmap='viridis')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.savefig("data2D.png")

# ------------------------ ITERATIVE TRAINING ---------------------

nqubits = nvariables
nlayers = 3
nruns = 5

print(y_data)
print(y_data.flatten())
print(x_data)

mse_list = []
r2_list=[]

start_time = datetime.datetime.now()
print(start_time)
for run in range(nruns):
    print(f"Running training {run+1}/{nruns}")
    print(f"Time running {datetime.datetime.now()-start_time}")
    model = ReuploadingU3(nqubits=nqubits, nlayers=nlayers, data_dimensionality=(nvariables,))
    # model = FourierReuploading(nqubits=nqubits, nlayers=nlayers, data_dimensionality=(1,))

    print(model.parameters)

    h = hamiltonians.Z(nqubits=nvariables)

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
    r2 = round(float(np.corrcoef(y_data, predictions)[0,1]**2),2)
    plt.figure(figsize=(5, 5 * 6 / 8))
    plt.plot(y_data, label="labels",color='tab:blue')
    plt.plot(predictions, label=f'predictions\nR2: {r2}',color='tab:orange')
    plt.legend()
    plt.savefig(f"u3_run_{run}.png")

    mse_list.append(mse(y_data, predictions))
    r2_list.append(r2)
mean_mse = np.mean(mse_list)
std_mse = np.std(mse_list)

mean_r2= np.mean(r2_list)
std_r2 = np.std(r2_list)

print(f"MSE estimation: {mean_mse} +- {std_mse}")
print(f"R2 estimation: {mean_r2} +- {std_r2}")

