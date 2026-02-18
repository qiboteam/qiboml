import math
from dataclasses import dataclass
from typing import Tuple
from numpy.typing import ArrayLike

from qibo import Circuit
from qibo.config import raise_error
from qiboml.differentiations.abstract import Differentiation


@dataclass
class PSR(Differentiation):
    """
    The Parameter Shift Rule differentiator. Especially useful for non analytical
    derivative calculation which, thus, makes it hardware compatible.
    """

    def _on_build(self):
        if math.prod(self.decoding.output_shape) != 1:
            raise_error(
                RuntimeError,
                "PSR differentiation works only for decoders with scalar outpus, i.e. expectation values.",
            )

    def evaluate(self, parameters: ArrayLike, wrt_inputs: bool = False):
        """
        Evaluate the gradient of the quantum circuit w.r.t its parameters, i.e. its rotation angles.
        Args:
            parameters (List[ArrayLike]): the parameters at which to evaluate the model, and thus the derivative.
            wrt_inputs (bool): whether to calculate the derivative with respect to, also, inputs (i.e. encoding angles)
        or not, by default ``False``.
        Returns:
            (ArrayLike): the calculated jacobian.
        """

        assert self._is_built, "Call .build(circuit, decoding) before evaluate()."

        circuits = []
        eigvals = []

        for i in range(self.nparams(wrt_inputs)):
            forward, backward, eigval = self.one_parameter_shift(
                parameters=parameters, parameter_index=i, wrt_inputs=wrt_inputs
            )
            circuits.extend([forward, backward])
            eigvals.append(eigval)

        # TODO: parallelize when decoding will support
        # the parallel execution of multiple circuits
        expvals = self.backend.cast(
            [self.decoding(circ) for circ in circuits], dtype=parameters.dtype
        )
        forwards = expvals[::2]
        backwards = expvals[1::2]
        eigvals = self.backend.reshape(
            self.backend.cast(eigvals, dtype=parameters.dtype), forwards.shape
        )

        return (forwards - backwards) * eigvals

    def one_parameter_shift(
        self, parameters: ArrayLike, parameter_index: int, wrt_inputs: bool = False
    ) -> Tuple[Circuit, Circuit, float]:
        """Compute one derivative of the decoding strategy w.r.t. a target parameter."""
        target_gates = (
            self.circuit.parametrized_gates
            if wrt_inputs
            else self.circuit.trainable_gates
        )
        gate = target_gates[parameter_index]
        generator_eigenval = gate.generator_eigenvalue()
        s = math.pi / (4 * generator_eigenval)

        tmp_params = self.backend.cast(parameters, copy=True, dtype=parameters[0].dtype)
        tmp_params = self.shift_parameter(tmp_params, parameter_index, s, self.backend)

        forward = self.circuit.copy(True)
        target_gates = (
            forward.parametrized_gates if wrt_inputs else forward.trainable_gates
        )
        # forward.set_parameters(tmp_params)
        for g, p in zip(target_gates, tmp_params):
            g.parameters = p
        forward._final_state = None

        tmp_params = self.backend.cast(parameters, copy=True, dtype=parameters[0].dtype)
        tmp_params = self.shift_parameter(tmp_params, parameter_index, -s, self.backend)

        backward = self.circuit.copy(True)
        target_gates = (
            backward.parametrized_gates if wrt_inputs else backward.trainable_gates
        )
        # backward.set_parameters(tmp_params)
        for g, p in zip(target_gates, tmp_params):
            g.parameters = p
        backward._final_state = None

        return forward, backward, generator_eigenval

    @staticmethod
    def shift_parameter(parameters, i, epsilon, backend):
        if backend.platform == "tensorflow":
            return backend.engine.stack(
                [parameters[j] + int(i == j) * epsilon for j in range(len(parameters))]
            )

        if backend.platform == "jax":
            parameters = parameters.at[i].set(parameters[i] + epsilon)
        else:
            parameters[i] = parameters[i] + epsilon

        return parameters
