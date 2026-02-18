from numpy.typing import ArrayLike

from qiboml.differentiations.abstract import Differentiation
from qiboml.models.decoding import Expectation


class Adjoint(Differentiation):
    """Adjoint differentiation following Algorithm 1 from Ref [1].

    Only :class:`qiboml.models.decoding.Expectation`. as decoding is supported
    and all parametrized_gates must have a gradient method returning the
    gradient of the single gate.

    References:
        1. T. Jones and J. Gacon, *Efficient calculation of gradients in classical
        simulations of variational quantum algorithms*, `arxiv:2009.02823 [quant-ph]
        <https://arxiv.org/abs/2009.02823>`_.
    """

    def evaluate(self, parameters: ArrayLike, wrt_inputs: bool = False) -> ArrayLike:
        """
        Evaluate the gradient of the quantum circuit w.r.t its parameters,
        *i.e.* its rotation angles.

        Args:
            parameters (List[ArrayLike]): the parameters at which to evaluate
                the model, and thus the derivative.
            wrt_inputs (bool): whether to calculate the derivate with respect
                to inputs or not, by default ``False``.

        Returns:
            ArrayLike: The calculated gradients.
        """

        assert (
            self._is_built
        ), "Call .build_differentiation(circuit, decoding) before evaluate()."

        assert isinstance(
            self.decoding, Expectation
        ), "Adjoint differentation supported only for Expectation."
        gate_list = (
            self.circuit.trainable_gates
            if not wrt_inputs
            else self.circuit.parametrized_gates
        )
        for g, p in zip(gate_list, parameters):
            g.parameters = p
        self.circuit._final_state = None

        gradients = []
        lam = self.backend.execute_circuit(self.circuit).state()
        nqubits = self.circuit.nqubits
        phi = lam
        lam = self.decoding.observable @ lam  # pylint: disable=E1101
        for gate in reversed(self.circuit.queue):
            phi = self.backend.apply_gate(gate.dagger(), phi, nqubits=nqubits)
            if gate in gate_list:
                mu = phi
                mu = self.backend.apply_gate(
                    gate.gradient(backend=self.backend), mu, nqubits=nqubits
                )
                gradients.append(
                    2 * self.backend.real(self.backend.engine.vdot(lam, mu))
                )
            lam = self.backend.apply_gate(gate.dagger(), lam, nqubits=nqubits)
        return self.backend.cast(gradients[::-1], dtype=parameters.dtype).reshape(
            -1, *self.decoding.output_shape
        )
