"""Parametric Quantum Circuit"""

from typing import Dict, List, Optional, Union

from numpy import array, ndarray
from qibo import Circuit
from qibo.config import raise_error
from qibo.gates import Gate
from qibo.hamiltonians import Hamiltonian

from qiboml.models.encodings import EncodingCircuit
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

    def add(self, gate: Gate):
        """
        Add a gate to the PQC.

        Args:
            gate (qibo.Gate): Qibo gate to be added to the PQC.
        """
        super().add(gate)
        if len(gate.parameters) != 0:
            self.parameters.extend(gate.parameters)
            self.nparams += gate.nparams

    def set_parameters(self, parameters: Union[List, ndarray]):
        """
        Set model parameters.

        Args:
            parameters (Union[List, ndarray]): new set of parameters to be set
                into the PQC.
        """
        self.parameters = parameters
        super().set_parameters(parameters)

    def setup(
        self,
        optimizer: Optimizer,
        loss: callable,
        observable: Union[Hamiltonian, List[Hamiltonian]],
        encoding_config: EncodingCircuit,
    ):
        """
        Compile the PQC to perform a training.

        Args:
            optimizer (qiboml.optimizers.Optimizer): optimizer to be used.
            loss (callable): loss function to be minimizer.
            observable (qibo.hamiltonians.Hamiltonian): observable, or list of
                observables, whose expectation value is used to compute predictions.
            encoding_config (qiboml.models.EncodingCircuit): encoding circuit, which
                is a Qibo circuit defined together with an encoding strategy.

        """
        self.optimizer = optimizer
        self.loss = loss
        self.observable = observable
        self.encoding_circuit = encoding_config
        self._compiled = True

    def fit(
        self,
        input_data: ndarray,
        output_data: ndarray,
        nshots: Optional[int] = None,
        options: Optional[Dict] = None,
    ):
        """
        Perform the PQC training according to the chosen trainig setup.

        Args:
            input_data (np.ndarray): input data to train on.
            output_data (np.ndarray): output data used as labels in the training process.
            nshots (Optional[int]): number of shots for circuit evaluations.
            options (Optional[Dict]): extra fit options eventually needed by the
                chosen optimizer.
        """

        if not self._compiled:
            raise_error(
                ValueError,
                "Please compile the model through the `PQC.setup` method to train it.",
            )

        if options is None:
            fit_options = {}
        else:
            fit_options = options

        def _loss(parameters, input_data, output_data):
            self.set_parameters(parameters)

            predictions = []
            for x in input_data:
                predictions.append(self.predict(input_datum=x, nshots=nshots))
            loss_value = self.loss(predictions, output_data)
            return loss_value

        results = self.optimizer.fit(
            initial_parameters=self.parameters,
            loss=_loss,
            args=(input_data, output_data),
            **fit_options
        )

        return results

    def predict(self, input_datum: Union[array, List, tuple], nshots: int = None):
        """
        Perform prediction associated to a single ``input_datum``.

        Args:
            input_datum (Union[array, List, tuple]): one single element of the
                input dataset.
            nshots (int): number of circuit execution to compute the prediction.
        """

        if not self._compiled:
            raise_error(
                ValueError,
                "Please compile the model through the `PQC.compile` method to perform predictions.",
            )

        encoding_state = self.encoding_circuit.inject_data(input_datum)().state()
        return self.observable.expectation(
            self(initial_state=encoding_state, nshots=nshots).state()
        )

    def predict_sample(self, input_data: ndarray, nshots: int = None):
        """
        Compute predictions for a set of data ``input_data``.

        Args:
            input_data (np.ndarray): input data.
            nshots (int): number of times the circuit is executed to compute the
                predictions.
        """

        predictions = []
        for x in input_data:
            predictions.append(self.predict(input_datum=x, nshots=nshots))

        return predictions
