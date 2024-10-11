import jax
import jax.numpy as jnp
import numpy as np
from qibo import parameter
from qibo.backends import construct_backend
from qibo.config import raise_error
from qibo.hamiltonians.abstract import AbstractHamiltonian
from torch.autograd import forward_ad

from qiboml import ndarray
from qiboml.models.abstract import QuantumCircuitLayer, _run_layers

"""
class PSR:

    def __init__(self):
        self.scale_factor = 1.0
        self.epsilon = 1e-2

    def evaluate(self, x: ndarray, layers: list[QuantumCircuitLayer], *parameters):
        backend = layers[-1].backend
        if backend.name == "pytorch":
            breakpoint()
        gradients = []
        index = 0
        for layer in layers:
            if not layer.has_parameters:
                continue
            parameters_bkup = [
                backend.cast(par, dtype=backend.precision, copy=True)
                for par in layer.parameters
            ]
            for i, param in enumerate(layer.parameters):
                grad = []
                param = backend.cast(param, backend.precision)
                for j in range(len(param)):
                    forward_backward = []
                    for shift in self._shift_parameters(param, j, self.epsilon):
                        params = list(layer.parameters)
                        params[i] = shift
                        layer.parameters = params
                        x_copy = x.clone()
                        for layer in layers:
                            x_copy = layer(x_copy)
                        forward_backward.append(x_copy)
                        layer.parameters = parameters_bkup
                    grad.append(
                        (forward_backward[0] - forward_backward[1]) * self.scale_factor
                    )
                gradients.append(backend.cast(grad))

        return gradients

    def _evaluate_for_parameter(self, x, layers, layer, index, parameters_bkup):
        outputs = []
        parameters_bkup = [backend.cast(par) for par in layer.parameters]
        for shift in self._shift_parameters(layer.parameters, index, self.epsilon):
            layer.parameters = shift
            outputs.append(_run_layers(x, layers, [l.parameters for l in layers]))
            layer.parameters = parameters_bkup
        return (outputs[0] - outputs[1]) * self.scale_factor

    @staticmethod
    def _shift_parameters(parameters: ndarray, index: int, epsilon: float):
        forward = parameters.clone()
        backward = parameters.clone()
        forward[index] += epsilon
        backward[index] -= epsilon
        return forward, backward
"""


class PSR:

    def __init__(self):
        self.scale_factor = 1.0
        self.epsilon = 1e-2

    def evaluate(self, x: ndarray, encoding, training, decoding, backend, *parameters):
        if decoding.output_shape != (1, 1):
            raise_error(
                NotImplementedError,
                "Parameter Shift Rule only supports expectation value decoding.",
            )
        x = encoding(x) + training
        gradients = []
        for i, param in enumerate(parameters):
            tmp_params = backend.cast(parameters, copy=True)
            tmp_params = PSR.shift_parameter(tmp_params, i, self.epsilon, backend)
            x.set_parameters(tmp_params)
            forward = decoding(x)
            tmp_params = PSR.shift_parameter(tmp_params, i, -2 * self.epsilon, backend)
            x.set_parameters(tmp_params)
            backward = decoding(x)
            gradients.append((forward - backward) * self.scale_factor)
        return gradients

    @staticmethod
    def shift_parameter(parameters, i, epsilon, backend):
        if backend.name == "tensorflow":
            return backend.tf.stack(
                [parameters[j] + int(i == j) * epsilon for j in range(len(parameters))]
            )
        elif backend.name == "jax":
            parameters.at[i].set(parameters[i] + epsilon)
        else:
            parameters[i] = parameters[i] + epsilon
        return parameters


class Jax:

    def __init__(self):
        self._input = None

    def evaluate(self, x: ndarray, encoding, training, decoding, backend, *parameters):
        self._input = x
        self.layers = layers
        parameters, indices = [], []
        for layer in layers:
            if layer.has_parameters:
                parameters.extend(layer.parameters.ravel())
                indices.append(len(parameters))
        parameters = jnp.asarray(parameters)
        gradients = jax.jacfwd(self._run)(parameters)
        return [
            gradients[:, :, i[0] : i[1]].squeeze(0).T
            for i in list(zip([0] + indices[:-1], indices))
        ]

    def _run(self, parameters):
        grouped_parameters = []
        left_index = right_index = 0
        for layer in self.layers:
            if layer.has_parameters:
                right_index += len(layer.parameters)
                grouped_parameters.append(parameters[left_index:right_index])
                left_index = right_index
        return _run_layers(self._input, self.layers, grouped_parameters)


def parameter_shift(
    hamiltonian,
    circuit,
    parameter_index,
    exec_backend,
    initial_state=None,
    scale_factor=1,
    nshots=None,
):
    """In this method the parameter shift rule (PSR) is implemented.
    Given a circuit U and an observable H, the PSR allows to calculate the derivative
    of the expected value of H on the final state with respect to a variational
    parameter of the circuit.
    There is also the possibility of setting a scale factor. It is useful when a
    circuit's parameter is obtained by combination of a variational
    parameter and an external object, such as a training variable in a Quantum
    Machine Learning problem. For example, performing a re-uploading strategy
    to embed some data into a circuit, we apply to the quantum state rotations
    whose angles are in the form: theta' = theta * x, where theta is a variational
    parameter and x an input variable. The PSR allows to calculate the derivative
    with respect of theta' but, if we want to optimize a system with respect its
    variational parameters we need to "free" this procedure from the x depencency.
    If the `scale_factor` is not provided, it is set equal to one and doesn't
    affect the calculation.
    If the PSR is needed to be executed on a real quantum device, it is important
    to set `nshots` to some integer value. This enables the execution on the
    hardware by calling the proper methods.

    Args:
        circuit (:class:`qibo.models.circuit.Circuit`): custom quantum circuit.
        hamiltonian (:class:`qibo.hamiltonians.Hamiltonian`): target observable.
            if you want to execute on hardware, a symbolic hamiltonian must be
            provided as follows (example with Pauli Z and ``nqubits=1``):
            ``SymbolicHamiltonian(np.prod([ Z(i) for i in range(1) ]))``.
        parameter_index (int): the index which identifies the target parameter
            in the ``circuit.get_parameters()`` list.
        initial_state (ndarray, optional): initial state on which the circuit
            acts. Default is ``None``.
        scale_factor (float, optional): parameter scale factor. Default is ``1``.
        nshots (int, optional): number of shots if derivative is evaluated on
            hardware. If ``None``, the simulation mode is executed.
            Default is ``None``.
        execution_backend (str): Qibo backend on which the circuits are executed.

    Returns:
        (float): Value of the derivative of the expectation value of the hamiltonian
            with respect to the target variational parameter.

    Example:

        .. testcode::

            import qibo
            import numpy as np
            from qibo import Circuit, gates, hamiltonians
            from qibo.derivative import parameter_shift

            # defining an observable
            def hamiltonian(nqubits = 1):
                m0 = (1/nqubits)*hamiltonians.Z(nqubits).matrix
                ham = hamiltonians.Hamiltonian(nqubits, m0)

                return ham

            # defining a dummy circuit
            def circuit(nqubits = 1):
                c = Circuit(nqubits = 1)
                c.add(gates.RY(q = 0, theta = 0))
                c.add(gates.RX(q = 0, theta = 0))
                c.add(gates.M(0))

                return c

            # initializing the circuit
            c = circuit(nqubits = 1)

            # some parameters
            test_params = np.random.randn(2)
            c.set_parameters(test_params)

            test_hamiltonian = hamiltonian()

            # running the psr with respect to the two parameters
            grad_0 = parameter_shift(circuit=c, hamiltonian=test_hamiltonian, parameter_index=0)
            grad_1 = parameter_shift(circuit=c, hamiltonian=test_hamiltonian, parameter_index=1)

    """

    # some raise_error
    if parameter_index > len(circuit.get_parameters()):
        raise_error(ValueError, """This index is out of bounds.""")

    if not isinstance(hamiltonian, AbstractHamiltonian):
        raise_error(
            TypeError,
            "hamiltonian must be a qibo.hamiltonians.Hamiltonian or qibo.hamiltonians.SymbolicHamiltonian object",
        )

    # getting the gate's type
    gate = circuit.associate_gates_with_parameters()[parameter_index]

    # getting the generator_eigenvalue
    generator_eigenval = gate.generator_eigenvalue()

    # defining the shift according to the psr
    s = np.pi / (4 * generator_eigenval)

    # saving original parameters and making a copy
    original = np.asarray(circuit.get_parameters()).copy()
    shifted = original.copy()

    # forward shift
    shifted[parameter_index] += s
    circuit.set_parameters(shifted)

    if nshots is None:
        # forward evaluation
        forward = hamiltonian.expectation(
            exec_backend.execute_circuit(
                circuit=circuit, initial_state=initial_state
            ).state()
        )

        # backward shift and evaluation
        shifted[parameter_index] -= 2 * s
        circuit.set_parameters(shifted)

        backward = hamiltonian.expectation(
            exec_backend.execute_circuit(
                circuit=circuit, initial_state=initial_state
            ).state()
        )

    # same but using expectation from samples
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

    # float() necessary to not return a 0-dim ndarray
    result = float(generator_eigenval * (forward - backward) * scale_factor)

    return result
