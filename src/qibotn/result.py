from copy import deepcopy
from dataclasses import dataclass
from typing import Union

from numpy import ndarray
from qibo.config import raise_error

from qibotn.backends.abstract import QibotnBackend


@dataclass
class TensorNetworkResult:
    """
    Object to store and process the output of a Tensor Network simulation of a quantum circuit.

    Args:
        nqubits (int): number of qubits involved in the simulation;
        backend (QibotnBackend): specific backend on which the simulation has been performed;
        measures (dict): measures (if performed) during the tensor network simulation;
        measured_probabilities (Union[dict, ndarray]): probabilities of the final state
            according to the simulation;
        prob_type (str): string identifying the method used to compute the probabilities.
            Especially useful in case the `QmatchateaBackend` is selected.
        statevector (ndarray): if computed, the reconstructed statevector.
    """

    nqubits: int
    backend: QibotnBackend
    measures: dict
    measured_probabilities: Union[dict, ndarray]
    prob_type: str
    statevector: ndarray

    def __post_init__(self):
        # TODO: define the general convention when using backends different from qmatchatea
        if self.measured_probabilities is None:
            self.measured_probabilities = {"default": self.measured_probabilities}

    def probabilities(self):
        """Return calculated probabilities according to the given method."""
        if self.prob_type == "U":
            measured_probabilities = deepcopy(self.measured_probabilities)
            for bitstring, prob in self.measured_probabilities[self.prob_type].items():
                measured_probabilities[self.prob_type][bitstring] = prob[1] - prob[0]
            probabilities = measured_probabilities[self.prob_type]
        else:
            probabilities = self.measured_probabilities
        return probabilities

    def frequencies(self):
        """Return frequencies if a certain number of shots has been set."""
        if self.measures is None:
            raise_error(
                ValueError,
                f"To access frequencies, circuit has to be executed with a given number of shots != None",
            )
        return self.measures

    def state(self):
        """Return the statevector if the number of qubits is less than 20."""
        if self.nqubits < 20:
            return self.statevector
        raise_error(
            NotImplementedError,
            f"Tensor network simulation cannot be used to reconstruct statevector for >= 20 .",
        )
