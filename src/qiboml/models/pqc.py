"""Parametric Quantum Circuit"""

from typing import Dict, List, Optional, Union

from numpy import ndarray
from qibo import Circuit
from qibo.config import raise_error
from qibo.hamiltonians import Hamiltonian

from qiboml.optimizers import Optimizer


class PQC(Circuit):
    """Parametric Quantum Circuit built on top of ``qibo.Circuit``."""

    def __init__(
        self,
        nqubits: int,
        accelerators: Optional[Dict] = None,
        density_matrix: bool = False,
        wire_names: Optional[Union[list, dict]] = None,
    ):
        super().__init__(nqubits, accelerators, density_matrix, wire_names)

        self.parameters = []
        self.nparams = 0

    def add(self, gate):
        super().add(gate)
        if len(gate.parameters) != 0:
            self.parameters.extend(gate.parameters)
            self.nparams += gate.nparams

    def compile(
        self,
        optimizer: Optimizer,
        loss: callable,
        observable: Union[Hamiltonian, List[Hamiltonian]],
        encoding_config: Circuit,
    ):
        """
        Compile the PQC to perform a training.

        Args:
            optimizer (qiboml.optimizers.Optimizer): optimizer to be used.
            loss (callable): loss function to be minimizer.
            observable (qibo.hamiltonians.Hamiltonian): observable, or list of
                observables, whose expectation value is used to compute predictions.
        """
        self.optimizer = optimizer
        self.loss = loss
        self.observable = observable
        self.encoding_circuit = encoding_config
        self.compiled = True

    def fit(
        self,
        x_data: ndarray,
        y_data: ndarray,
        nshots: Optional[int] = None,
        options: Optional[Dict] = None,
    ):
        """Perform the PQC training."""

        if not self.compiled:
            raise_error(
                ValueError,
                "Please compile the model through the `PQC.compile` method to train it.",
            )

        if options is None:
            fit_options = {}
        else:
            fit_options = options

        def _loss(parameters, x_data, y_data):
            self.set_parameters(parameters)

            predictions = []
            for x in x_data:
                predictions.append(self.predict(x=x, nshots=nshots))
            loss_value = self.loss(predictions, y_data)
            return loss_value

        results = self.optimizer.fit(
            initial_parameters=self.parameters,
            loss=_loss,
            args=(x_data, y_data),
            **fit_options
        )

        return results

    def predict(self, x: ndarray, nshots: int = None):
        """Perform prediction associated to a single input data ``x``."""

        if not self.compiled:
            raise_error(
                ValueError,
                "Please compile the model through the `PQC.compile` method to perform predictions.",
            )

        print(self.compiled)

        encoding_state = self.encoding_circuit.inject_data(x)().state()
        return self.observable.expectation(
            self(initial_state=encoding_state, nshots=nshots).state()
        )
