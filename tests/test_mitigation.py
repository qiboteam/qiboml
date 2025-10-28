from copy import deepcopy

import pytest
from qibo import transpiler
from qibo.backends import NumpyBackend
from qibo.hamiltonians import XXZ, SymbolicHamiltonian
from qibo.noise import NoiseModel, PauliError
from qibo.transpiler.pipeline import Passes
from qibo.transpiler.unroller import NativeGates, Unroller

from qiboml.models.ansatze import hardware_efficient
from qiboml.models.decoding import Expectation
from qiboml.operations.differentiation import PSR

from .utils import set_seed

BACKENDS = [NumpyBackend()]


def build_noise_model(
    nqubits: int,
    local_pauli_noise_prob: float,
):
    """Costruct noise model as a local Pauli noise channel + readout noise."""
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


def train_vqe(frontend, backend, model, epochs):
    """Implement training procedure given interface."""

    if frontend.__name__ == "qiboml.interfaces.pytorch":
        optimizer = frontend.torch.optim.Adam(model.parameters(), lr=0.1)
        for _ in range(epochs):
            optimizer.zero_grad()
            cost = model()
            print(cost)
            cost.backward()
            optimizer.step()
        return backend.cast(cost.item(), dtype="double")

    elif frontend.__name__ == "qiboml.interfaces.keras":
        frontend.tf.keras.backend.set_floatx("float64")
        optimizer = frontend.keras.optimizers.Adam(learning_rate=0.3)
        for _ in range(epochs):
            with frontend.tf.GradientTape() as tape:
                cost = model()
            gradients = tape.gradient(cost, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return backend.cast(cost, dtype="double")


@pytest.mark.parametrize("backend", BACKENDS)
# @pytest.mark.parametrize("mitigation_method", ["ICS", "CDR"])
@pytest.mark.parametrize("mitigation_method", ["CDR"])
def test_rtqem(frontend, backend, mitigation_method):
    nqubits = 2
    nlayers = 4
    nshots = 10000

    seed = 1
    set_seed(frontend, seed)
    backend.set_seed(seed)

    # We build a trainable circuit
    vqe = hardware_efficient(nqubits=nqubits, nlayers=nlayers, seed=seed)
    from qibo.symbols import Z

    obs = SymbolicHamiltonian(Z(0) * Z(1), backend=backend)
    target_energy = min(obs.eigenvalues())

    noise_model = build_noise_model(nqubits=nqubits, local_pauli_noise_prob=0.003)

    # First we build a model with noise and without mitigation
    noisy_decoding = Expectation(
        nqubits=nqubits,
        observable=obs,
        nshots=nshots,
        backend=backend,
        noise_model=noise_model,
        density_matrix=True,
    )

    noisy_model = frontend.QuantumModel(
        circuit_structure=deepcopy(vqe),
        decoding=noisy_decoding,
        differentiation=PSR,
    )
    print("-------> Noisy")
    noisy_result = train_vqe(
        frontend=frontend,
        backend=backend,
        model=noisy_model,
        epochs=10,
    )

    mitigation_config = {
        "threshold": 3e-1,
        "method": mitigation_method,
        "method_kwargs": {"n_training_samples": 50, "nshots": 10000},
        "min_iterations": 100,
    }

    # Then we build a decoding with error mitigation
    mit_decoding = Expectation(
        nqubits=nqubits,
        observable=obs,
        nshots=nshots,
        backend=backend,
        noise_model=noise_model,
        density_matrix=True,
        mitigation_config=mitigation_config,
    )

    mit_model = frontend.QuantumModel(
        circuit_structure=deepcopy(vqe),
        decoding=mit_decoding,
        differentiation=PSR,
    )
    print("-------> Noise-Mitigated")
    mit_result = train_vqe(
        frontend=frontend,
        backend=backend,
        model=mit_model,
        epochs=10,
    )

    assert abs(mit_result - target_energy) < abs(noisy_result - target_energy)


def test_custom_map(frontend):
    seed = 42
    set_seed(frontend=frontend, seed=seed)

    # We build a trainable circuit
    nqubits = 1
    nlayers = 2
    vqe = hardware_efficient(nqubits=nqubits, nlayers=nlayers, seed=seed)

    mitigation_config = {
        "real_time": True,
        "method": "CDR",
        "method_kwargs": {
            "n_training_samples": 70,
            "model": lambda x, a, b: a * x + b,
        },
    }

    # Then we build a decoding with error mitigation
    mit_decoding = Expectation(
        nqubits=1,
        nshots=1024,
        backend=NumpyBackend(),
        noise_model=build_noise_model(nqubits=1, local_pauli_noise_prob=0.03),
        density_matrix=True,
        mitigation_config=mitigation_config,
    )

    initial_popt = mit_decoding.mitigator._mitigation_map_popt

    mit_model = frontend.QuantumModel(
        circuit_structure=deepcopy(vqe),
        decoding=mit_decoding,
        differentiation=PSR,
    )

    _ = train_vqe(
        frontend=frontend,
        backend=NumpyBackend(),
        model=mit_model,
        epochs=2,
    )

    diff = mit_decoding.mitigator._mitigation_map_popt != initial_popt
    assert diff.any()
