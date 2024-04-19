from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


def plot_loss_wrt_parameter(
    model,
    parameter_index: int,
    loss_function: callable,
    filename: Optional[str] = None,
    label: str = "",
    color: Optional[str] = None,
    return_losses: bool = False,
    alpha: float = 1,
):

    parameters = model.parameters
    angles = np.linspace(0, 2 * np.pi, 200)

    loss_values = []

    for angle in angles:
        parameters[parameter_index] = angle
        model.set_parameters(parameters)
        loss_values.append(loss_function(parameters, model))

    plt.plot(angles, loss_values, label=label, color=color, alpha=alpha)
    plt.xlabel(f"Parameter {parameter_index}")
    plt.ylabel("Loss")
    plt.legend()

    if filename:
        plt.savefig(f"{filename}.png")

    if return_losses:
        return loss_values
