#!/usr/bin/env python3
import json
import os
import time

import click
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from qibo import Circuit, construct_backend, gates, set_backend
from qibo.noise import NoiseModel, PauliError
from scipy.stats import median_abs_deviation

from qiboml.interfaces.pytorch import QuantumModel
from qiboml.models.decoding import Expectation
from qiboml.models.encoding import PhaseEncoding
from qiboml.operations import differentiation


def build_noise_model(nqubits: int, local_pauli_noise_prob: float):
    """Construct a local Pauli noise channel + readout noise model."""
    noise_model = NoiseModel()
    for q in range(nqubits):
        noise_model.add(
            PauliError(
                [
                    ("X", local_pauli_noise_prob),
                    ("Y", local_pauli_noise_prob),
                    ("Z", local_pauli_noise_prob),
                ]
            ),
            qubits=q,
        )
    return noise_model


def trainable_layer(nqubits: int, density_matrix: bool):
    """Return a new trainable circuit each call."""
    circuit = Circuit(nqubits=nqubits, density_matrix=density_matrix)
    for q in range(nqubits):
        circuit.add(gates.RY(q, np.random.uniform(-np.pi, np.pi)))
        circuit.add(gates.RZ(q, np.random.uniform(-np.pi, np.pi)))
    return circuit


@click.command()
@click.option("--backend", default="qiboml", help="Which Qibo backend to use.")
@click.option("--platform", default="pytorch", help="Which platform to use.")
@click.option(
    "--differentiation_rule", default=None, help="Which differentiation rule to use."
)
@click.option(
    "--with_noise", default=False, type=bool, help="If we want the noise model."
)
@click.option(
    "--with_mitigation",
    default=False,
    type=bool,
    help="If we want real time error mitigation.",
)
@click.option(
    "-n",
    "--n_runs",
    type=int,
    default=5,
    help="Number of repeat training runs for uncertainty estimation.",
)
@click.option(
    "--nepochs", type=int, default=30, help="Number of training epochs per run."
)
@click.option(
    "--nshots", type=int, default=None, help="Number of training epochs per run."
)
@click.option(
    "--seed", type=int, default=0, help="Base random seed for reproducibility."
)
def main(
    backend,
    platform,
    differentiation_rule,
    with_noise,
    with_mitigation,
    n_runs,
    nepochs,
    nshots,
    seed,
):
    """
    Train a simple quantum model multiple times, collect predictions and loss histories,
    compute median ± MAD, and save results under results/{backend}_{platform}_{differentiation_rule}.
    """

    params = locals().copy()

    # 1) Set base seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 2) Construct results directory
    custom_name = f"{backend}_{platform}_{differentiation_rule}_shots{nshots}_noise{with_noise}_mit{with_mitigation}_runs{n_runs}"
    results_dir = os.path.join("results", custom_name)
    os.makedirs(results_dir, exist_ok=True)

    # 3) Prepare exec_backend
    if platform is None:
        exec_backend = construct_backend(backend)
        set_backend(backend)
    else:
        exec_backend = construct_backend(backend, platform=platform)
        set_backend(backend, platform=platform)

    nqubits = 1
    nlayers = 4

    # 4) Prepare the training dataset once (same for all runs)
    def f(x):
        return 1 * torch.sin(x) ** 2 - 0.3 * torch.cos(x)

    num_samples = 20
    x_all = torch.linspace(0, 2 * np.pi, num_samples, dtype=torch.float64).unsqueeze(1)
    perm = torch.randperm(x_all.size(0))
    x_train = x_all[perm]
    y_train = f(x_train)
    # Normalize y to [-1, 1]
    y_train = 2 * ((y_train - y_train.min()) / (y_train.max() - y_train.min()) - 0.5)

    # Precompute sorting for consistent x order in plotting
    _, sorted_idx = torch.sort(x_train.squeeze(), dim=0)
    x_train_sorted = x_train[sorted_idx]
    y_train_sorted = y_train[sorted_idx]

    # 5) Noise and mitigation configuration
    if with_noise:
        noise_model = build_noise_model(nqubits, local_pauli_noise_prob=0.01)
        density_matrix = True
    else:
        noise_model = None
        density_matrix = False

    if with_mitigation:
        mitigation_config = {
            "real_time": True,
            "method": "CDR",
            "method_kwargs": {"n_training_samples": 50},
        }
    else:
        mitigation_config = None

    # 6) Embedding circuit (fixed)
    encoding_circ = PhaseEncoding(
        nqubits=nqubits,
        encoding_gate=gates.RX,
        density_matrix=density_matrix,
    )

    # 7) Function to instantiate a fresh QuantumModel
    def create_model():
        # Build circuit structure: [encoding, trainable] × nlayers
        circ_structure = []
        for _ in range(nlayers):
            circ_structure.append(encoding_circ)
            circ_structure.append(trainable_layer(nqubits, density_matrix))
        # Decoding (with noise/mitigation)
        decoding_circ = Expectation(
            nqubits=nqubits,
            density_matrix=density_matrix,
            backend=exec_backend,
            nshots=nshots,
            noise_model=noise_model,
            mitigation_config=mitigation_config,
        )
        # Differentiation rule
        if differentiation_rule not in [None, "PSR", "Jax"]:
            raise ValueError(f"Unknown differentiation rule: {differentiation_rule}")
        diff_rule = None
        if differentiation_rule is not None:
            diff_rule_cls = getattr(differentiation, differentiation_rule)
            diff_rule = diff_rule_cls()

        return QuantumModel(
            circuit_structure=circ_structure,
            decoding=decoding_circ,
            differentiation=diff_rule,
        )

    # 8) Containers to hold data across runs
    all_runs_preds = np.zeros((n_runs, num_samples), dtype=np.float32)
    loss_histories = np.zeros((n_runs, nepochs), dtype=np.float32)
    times_per_run = np.zeros(n_runs, dtype=np.float64)

    # 9) Training loop repeated
    for run_idx in range(n_runs):
        current_seed = seed + run_idx
        torch.manual_seed(current_seed)
        np.random.seed(current_seed)

        print(f"\n=== Starting run {run_idx + 1}/{n_runs} (seed={current_seed}) ===")

        model = create_model()
        optimizer = optim.Adam(model.parameters(), lr=0.1)
        criterion = nn.MSELoss()

        # Time the training for this run
        start_time = time.time()

        # Training epochs
        for epoch in range(nepochs):
            optimizer.zero_grad()
            # Forward pass: batch over x_train
            y_pred_batch = torch.stack([model(x) for x in x_train])
            y_pred_batch = y_pred_batch.squeeze(-1)
            loss = criterion(y_pred_batch, y_train)
            loss.backward()
            optimizer.step()

            # Record loss
            loss_histories[run_idx, epoch] = loss.item()

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(
                    f" Run {run_idx + 1} | Epoch {epoch + 1}/{nepochs} | Loss: {loss.item():.6f}"
                )

        # Record elapsed time for this run
        times_per_run[run_idx] = time.time() - start_time

        # After training, get predictions on sorted x
        with torch.no_grad():
            preds_sorted = []
            for x_val in x_train_sorted:
                preds_sorted.append(model(x_val).item())
            all_runs_preds[run_idx, :] = np.array(preds_sorted, dtype=np.float32)

    # 10) Compute median and median absolute deviation across runs
    median_preds = np.median(all_runs_preds, axis=0)
    mad_preds = median_abs_deviation(all_runs_preds, axis=0)
    times_median = np.median(times_per_run)
    times_mad = median_abs_deviation(times_per_run, axis=0)

    # Compute median & MAD of loss history across runs
    median_loss = np.median(loss_histories, axis=0)
    mad_loss = median_abs_deviation(loss_histories, axis=0)

    # 11) Save numpy arrays
    np.save(os.path.join(results_dir, "all_runs_preds.npy"), all_runs_preds)
    np.save(os.path.join(results_dir, "all_runs_loss_histories.npy"), loss_histories)
    np.save(os.path.join(results_dir, "times_per_run.npy"), times_per_run)

    # 12) Prepare summary dictionary
    params.update(
        {
            "model_diff_rule": str(model.differentiation),
            "times_median": float(times_median),
            "times_mad": float(times_mad),
            "median_preds": median_preds.tolist(),
            "mad_preds": mad_preds.tolist(),
            "median_loss": median_loss.tolist(),
            "mad_loss": mad_loss.tolist(),
        }
    )

    # 13) Save summary as JSON
    with open(os.path.join(results_dir, "summary.json"), "w") as jf:
        json.dump(params, jf, indent=2)

    # 14) Plot targets, median predictions, and uncertainty band (save as PDF)
    plt.figure(figsize=(5, 4), dpi=120)
    x_np = x_train_sorted.squeeze().numpy()
    target_np = y_train_sorted.squeeze().numpy()

    # Plot target function
    plt.plot(
        x_np,
        target_np,
        marker=".",
        markersize=8,
        color="blue",
        label="Targets",
        alpha=0.7,
    )

    # Plot median prediction
    plt.plot(
        x_np,
        median_preds,
        marker=".",
        markersize=8,
        color="red",
        label="Median Prediction",
        alpha=0.7,
    )

    # Shaded uncertainty band: median ± MAD
    lower = median_preds - mad_preds
    upper = median_preds + mad_preds
    plt.fill_between(
        x_np,
        lower,
        upper,
        color="orange",
        alpha=0.3,
        label="Median ± MAD",
    )

    plt.xlabel(r"$x$")
    plt.ylabel(r"$f(x)$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "prediction_with_uncertainty.pdf"))
    plt.close()

    # 15) Plot loss history median ± MAD (save as PDF)
    plt.figure(figsize=(5, 4), dpi=120)
    epochs_axis = np.arange(1, nepochs + 1)

    # Plot median loss
    plt.plot(
        epochs_axis,
        median_loss,
        marker=".",
        markersize=8,
        color="red",
        label="Median Loss",
        alpha=0.7,
    )

    # Shaded uncertainty band: median ± MAD
    loss_lower = median_loss - mad_loss
    loss_upper = median_loss + mad_loss
    plt.fill_between(
        epochs_axis,
        loss_lower,
        loss_upper,
        color="orange",
        alpha=0.3,
        label="Median ± MAD",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "loss_history_with_uncertainty.pdf"))
    plt.close()


if __name__ == "__main__":
    main()
