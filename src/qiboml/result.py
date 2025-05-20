from dataclasses import dataclass
from typing import List, Union

from qibo.result import CircuitResult, MeasurementOutcomes, QuantumState


@dataclass
class BatchedResult:

    results: List[Union[CircuitResult, QuantumState, MeasurementOutcomes]]
    _batchsize: int = None

    def __post_init__(self):
        self._batchsize = len(self.results)

    def probabilities(self, qubits):
        probs = [res.probabilities(qubits) for res in self.results]
        probs = self.backend.np.vstack(probs)
        return self.backend.np.reshape(probs, (self._batchsize, probs.shape[-1]))

    def state(self):
        states = [res.state() for res in self.results]
        shape = states[0].shape
        states = self.backend.np.vstack(states)
        return self.backend.np.reshape(states, (self._batchsize, shape))

    def samples(self):
        samples = [res.samples() for res in self.results]
        shape = samples[0].shape
        samples = self.backend.np.vstack(samples)
        return self.backend.np.reshape(samples, (self._batchsize, shape))

    def frequencies(self):
        return [res.frequencies() for res in self.results]

    @property
    def backend(self):
        return self.results[0].backend
