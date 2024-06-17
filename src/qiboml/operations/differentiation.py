from typing import Optional, Union

import jax
import numpy as np
import qibo
import qibo.backends
from qibo.config import raise_error
from qibo.hamiltonians.abstract import AbstractHamiltonian

from qiboml.operations import expectation


def _one_parameter_shift(
    hamiltonian: qibo.hamiltonians.Hamiltonian,
    circuit: qibo.Circuit,
    parameter_index: int,
    initial_state: Optional[Union[np.ndarray, qibo.Circuit]],
    nshots: int,
    exec_backend: qibo.backends.Backend,
):
    """
    Helper method to compute the derivative of the expectation value of
    ``hamiltonian`` over the state we get executing ``circuit`` starting from
    ``initial_state`` w.r.t. to a target ``parameter_index``.
    """

    if parameter_index > len(circuit.get_parameters()):
        raise_error(ValueError, """This index is out of bounds.""")

    if not isinstance(hamiltonian, AbstractHamiltonian):
        raise_error(
            TypeError,
            "hamiltonian must be a qibo.hamiltonians.Hamiltonian or qibo.hamiltonians.SymbolicHamiltonian object",
        )

    gate = circuit.associate_gates_with_parameters()[parameter_index]
    generator_eigenval = gate.generator_eigenvalue()

    s = np.pi / (4 * generator_eigenval)

    original = np.asarray(circuit.get_parameters()).copy()
    shifted = original.copy()

    shifted[parameter_index] += s
    circuit.set_parameters(shifted)

    if nshots is None:

        forward = expectation._exact(hamiltonian, circuit, initial_state, exec_backend)

        shifted[parameter_index] -= 2 * s
        circuit.set_parameters(shifted)

        backward = expectation._exact(hamiltonian, circuit, initial_state, exec_backend)

    else:
        forward = exec_backend.execute_circuit(
            circuit=circuit, initial_state=initial_state, nshots=nshots
        ).expectation_from_samples(hamiltonian)

        shifted[parameter_index] -= 2 * s
        circuit.set_parameters(shifted)

        backward = exec_backend.execute_circuit(
            circuit=circuit, initial_state=initial_state, nshots=nshots
        ).expectation_from_samples(hamiltonian)

    circuit.set_parameters(original)

    return float(generator_eigenval * (forward - backward))


def parameter_shift(
    hamiltonian: qibo.hamiltonians.Hamiltonian,
    circuit: qibo.Circuit,
    exec_backend: qibo.backends.Backend,
    initial_state: Optional[Union[np.ndarray, qibo.Circuit]] = None,
    nshots: int = None,
):
    """
    Compute the gradient of the expectation value of ``hamiltonian`` over the
    state we get executing ``circuit`` over ``initial_state`` with a certain
    ``nshots`` w.r.t. the parameters of the circuit via parameter-shift
    rule (https://arxiv.org/abs/1811.11184).
    The number of shots can be set to be ``None`` in case of exact simulation,
    otherwise an integer number of shots has to be provided.

    Args:
        circuit (:class:`qibo.models.circuit.Circuit`): custom quantum circuit.
        hamiltonian (:class:`qibo.hamiltonians.Hamiltonian`): target observable.
            if you want to execute on hardware, a symbolic hamiltonian must be
            provided as follows (example with Pauli Z and ``nqubits=1``):
            ``SymbolicHamiltonian(np.prod([ Z(i) for i in range(1) ]))``.
        execution_backend (qibo.backends.Backend): Qibo backend on which the
            circuits are executed.
        initial_state (ndarray, optional): initial state on which the circuit
            acts. Default is ``None``.
        nshots (int, optional): number of shots if derivative is evaluated on
            hardware. If ``None``, the simulation mode is executed.
            Default is ``None``.

    Returns:
        (float): Gradient of the expectation value of the hamiltonian
            with respect to circuit's variational parameters.

    """
    return [
        _one_parameter_shift(
            hamiltonian=hamiltonian,
            circuit=circuit,
            parameter_index=i,
            initial_state=initial_state,
            nshots=nshots,
            exec_backend=exec_backend,
        )
        for i in range(len(circuit.get_parameters()))
    ]


def symbolical(
    hamiltonian: qibo.hamiltonians.Hamiltonian,
    circuit: qibo.Circuit,
    exec_backend: qibo.backends.Backend,
    initial_state: Optional[Union[np.ndarray, qibo.Circuit]] = None,
):
    """
    Compute the gradient of the expectation value of ``hamiltonian`` over the
    state we get executing ``circuit`` over ``initial_state`` with a certain
    ``nshots`` w.r.t. the parameters of the circuit using TensorFlow automatic
    differentiation.

    Args:
        circuit (:class:`qibo.models.circuit.Circuit`): custom quantum circuit.
        hamiltonian (:class:`qibo.hamiltonians.Hamiltonian`): target observable.
            if you want to execute on hardware, a symbolic hamiltonian must be
            provided as follows (example with Pauli Z and ``nqubits=1``):
            ``SymbolicHamiltonian(np.prod([ Z(i) for i in range(1) ]))``.
        execution_backend (qibo.backends.Backend): Qibo backend on which the
            circuits are executed.
        initial_state (ndarray, optional): initial state on which the circuit
            acts. Default is ``None``.

    Returns:
        (float): Gradient of the expectation value of the hamiltonian
            with respect to circuit's variational parameters.

    """
    import tensorflow as tf  # pylint: disable=import-error

    # TODO: how to fix this at lower level?
    circuit_parameters = tf.Variable(circuit.get_parameters(), dtype="float64")

    with tf.GradientTape() as tape:
        circuit.set_parameters(circuit_parameters)
        expval = expectation._exact(hamiltonian, circuit, initial_state, exec_backend)

    gradients = tape.gradient(expval, circuit_parameters)

    return tf.squeeze(gradients)


def symbolical_with_jax(
    hamiltonian: qibo.hamiltonians.Hamiltonian,
    circuit: qibo.Circuit,
    exec_backend,
    initial_state: Optional[Union[np.ndarray, qibo.Circuit]] = None,
):
    def _expectation(params):
        params = jax.numpy.array(params)
        circuit.set_parameters(params)
        return expectation._exact(hamiltonian, circuit, initial_state, exec_backend)

    return jax.grad(_expectation)(circuit.get_parameters())
