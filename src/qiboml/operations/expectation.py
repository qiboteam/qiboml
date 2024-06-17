"""Compute expectation values of target observables with the freedom of setting any qibo's backend."""

from typing import List, Optional, Union

import qibo
from qibo.backends import Backend, NumpyBackend
from qibo.config import raise_error

from qiboml.backends import JaxBackend, TensorflowBackend


def expectation(
    observable: qibo.hamiltonians.Hamiltonian,
    circuit: qibo.Circuit,
    initial_state: Optional[Union[List, qibo.Circuit]] = None,
    nshots: int = None,
    exec_backend: Backend = NumpyBackend(),
    differentiation_rule: Optional[callable] = None,
):
    """
    Compute the expectation value of ``observable`` over the state obtained by
    executing ``circuit`` starting from ``initial_state``. The final state is
    reconstructed from ``nshots`` execution of ``circuit`` on the selected ``backend``.
    In addition, a differentiation rule can be set, which is going to be integrated
    within the used high-level framework. For example, if TensorFlow is used
    in the user code and one parameter shift rule is selected as differentiation
    rule, the expectation value is computed informing the TensorFlow graph to
    use as gradient the output of the parameter shift rule executed on the selected
    backend.

    Args:
        observable (qibo.Hamiltonian): the observable whose expectation value has
            to be computed.
        circuit (qibo.Circuit): quantum circuit returning the final state over which
            the expectation value of ``observable`` is computed.
        initial_state (Optional[Union[List, qibo.Circuit]]): initial state on which
            the quantum circuit is applied.
        nshots (int): number of times the quantum circuit is executed. Increasing
            the number of shots will reduce the variance of the estimated expectation
            value while increasing the computational cost of the operation.
        exec_backend (qibo.backend.Backend): backend on which the circuit
            is executed. This same backend is used if the chosen differentiation
            rule makes use of expectation values.
        differentiation_rule (Optional[callable]): the chosen differentiation
            rule. It can be selected among the methods implemented in
            ``qiboml.differentiation``.
    """

    # read the frontend user choice
    frontend = observable.backend

    return frontend.calculate_expval(
        observable=observable,
        circuit=circuit,
        initial_state=initial_state,
        nshots=nshots,
        exec_backend=exec_backend,
        differentiation_rule=differentiation_rule,
    )


def _exact(observable, circuit, initial_state, exec_backend):
    """Helper function to compute exact expectation values."""
    return exec_backend.calculate_expectation_state(
        hamiltonian=exec_backend.cast(observable.matrix),
        state=exec_backend.execute_circuit(
            circuit=circuit, initial_state=initial_state
        ).state(),
        normalize=False,
    )


def _with_shots(observable, circuit, initial_state, nshots, exec_backend):
    """Helper function to compute expectation values from samples."""
    return exec_backend.execute_circuit(
        circuit=circuit, initial_state=initial_state, nshots=nshots
    ).expectation_from_samples(observable)
