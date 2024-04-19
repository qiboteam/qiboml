import matplotlib.pyplot as plt
import numpy as np
import qibo
from qibo import gates, hamiltonians

from qiboml.models.reuploading.fourier import FourierReuploading
from qiboml.models.reuploading.u3 import ReuploadingU3
from qiboml.visualization.training_landscape import plot_loss_wrt_parameter

qibo.set_backend("numpy")

nqubits = 8
nlayers = 10
parameter_index = 7

model = ReuploadingU3(nqubits=nqubits, nlayers=nlayers, data_dimensionality=(nqubits,))


def loss_function(parameters, model):
    model.set_parameters(parameters)
    model.inject_data((0.9,) * nqubits)
    ham = hamiltonians.Z(nqubits=nqubits)
    return ham.expectation(model.circuit().state())


plot_loss_wrt_parameter(
    model=model,
    parameter_index=parameter_index,
    loss_function=loss_function,
    label="Original",
    color="black",
)


losses = []

for i in range(10):
    print(f"Iteration {i}/{10}")
    perturbated_circuit = model.perturbated_circuit(ngates=1)
    model.circuit = perturbated_circuit

    loss_values = plot_loss_wrt_parameter(
        model=model,
        parameter_index=parameter_index,
        loss_function=loss_function,
        filename=f"perturbated_param_{parameter_index}",
        return_losses=True,
        color="royalblue",
        alpha=0.4,
    )

    losses.append(loss_values)


plt.plot(
    np.linspace(0, 2 * np.pi, 200),
    np.mean(np.array(losses), axis=0),
    color="black",
    ls="--",
    lw=1.5,
    label="Average",
)
plt.legend()
plt.savefig(f"perturbated_param_{parameter_index}.png")
