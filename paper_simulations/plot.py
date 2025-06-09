import numpy as np 
import matplotlib.pyplot as plt 
import scienceplots
from scipy.stats import median_abs_deviation

plt.style.use('science')

COLORS = ["red", "orange", "blue", "green"]

noisy_preds = np.load(
    "results/numpy_pytorch_None_shots1024_noiseTrue_mitFalse_runs10/all_runs_preds.npy"
)
mit_preds = np.load(
    "results/numpy_pytorch_None_shots1024_noiseTrue_mitTrue_runs10/all_runs_preds.npy"
)
shots_preds = np.load(
    "results/numpy_pytorch_None_shots1024_noiseFalse_mitFalse_runs10/all_runs_preds.npy"
)
exact_preds = np.load(
    "results/qiboml_pytorch_None_shotsNone_noiseFalse_mitFalse_runs10/all_runs_preds.npy"
)

def plot_results(x_np, target_np, median_preds, mad_preds, labels, save_as, legend=True):
    # 14) Plot targets, median predictions, and uncertainty band (save as PDF)
    plt.figure(figsize=(10 * 0.5 , 10 * 0.5 * 6/8), dpi=120)

    # Plot target function
    plt.plot(
        x_np,
        target_np,
        marker=".",
        markersize=10,
        color="black",
        label="Targets",
        ls="--",
        alpha=0.7,
    )

    for i in range(len(median_preds)):
        # Plot median prediction
        plt.plot(
            x_np,
            median_preds[i],
            marker=".",
            markersize=10,
            color=COLORS[i],
            label=labels[i],
            alpha=0.7,
        )

        # Shaded uncertainty band: median ± MAD
        lower = median_preds[i] - mad_preds[i]
        upper = median_preds[i] + mad_preds[i]
        plt.fill_between(
            x_np,
            lower,
            upper,
            color=COLORS[i],
            alpha=0.3,
        )

    plt.xlabel(r"$x$", fontsize=12)
    plt.ylabel(r"$f(x)$", fontsize=12)
    if legend:
        plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{save_as}.pdf", bbox_inches="tight")
    plt.close()

def f(x):
    return np.sin(x) ** 2 - 0.3 * np.cos(x)

x = np.linspace(0, 2*np.pi, 20)
y = f(x)
y = 2 * ( (y - min(y)) / (max(y) - min(y)) - 0.5 )


plot_results(
    x_np=x, 
    target_np=y, 
    median_preds=[
        np.median(noisy_preds, axis=0),
        np.median(mit_preds, axis=0),
        np.median(shots_preds, axis=0),
        np.median(exact_preds, axis=0),
    ], 
    mad_preds=[
        median_abs_deviation(noisy_preds, axis=0),
        median_abs_deviation(mit_preds, axis=0),
        median_abs_deviation(shots_preds, axis=0),
        median_abs_deviation(exact_preds, axis=0),
    ],
    labels=[
        "Noisy",
        "Mitigated",
        "Noiseless",
        "Exact simulation"
    ],
    save_as="noisy-mit"
)



