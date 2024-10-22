from dataclasses import dataclass
from typing import Union

from qibo import Circuit, gates
from qibo.backends import Backend, _check_backend
from qibo.config import raise_error
from qibo.hamiltonians import Hamiltonian

from qiboml import ndarray


@dataclass
class QuantumDecoding:

    nqubits: int
    qubits: list[int] = None
    nshots: int = 1000
    analytic: bool = True
    backend: Backend = None
    _circuit: Circuit = None

    def __post_init__(self):
        if self.qubits is None:
            self.qubits = list(range(self.nqubits))
        self._circuit = Circuit(self.nqubits)
        self.backend = _check_backend(self.backend)
        self._circuit.add(gates.M(*self.qubits))

    def __call__(self, x: Circuit) -> "CircuitResult":
        return self.backend.execute_circuit(x + self._circuit, nshots=self.nshots)

    @property
    def circuit(
        self,
    ):
        return self._circuit

    def set_backend(self, backend):
        self.backend = backend

    @property
    def output_shape(self):
        raise_error(NotImplementedError)


@dataclass
class Probabilities(QuantumDecoding):

    def __call__(self, x: Circuit) -> ndarray:
        return super().__call__(x).probabilities()

    @property
    def output_shape(self):
        return (1, 2**self.nqubits)


@dataclass
class Expectation(QuantumDecoding):

    observable: Union[ndarray, Hamiltonian] = None
    analytic: bool = False

    def __post_init__(self):
        if self.observable is None:
            raise_error(
                RuntimeError,
                "Please provide an observable for expectation value calculation.",
            )
        super().__post_init__()

    def __call__(self, x: Circuit) -> ndarray:
        if self.analytic:
            return self.observable.expectation(
                super().__call__(x).state(),
            ).reshape(1, 1)
        else:
            return self.observable.expectation_from_samples(
                super().__call__(x).frequencies(),
                qubit_map=self.qubits,
            ).reshape(1, 1)

    @property
    def output_shape(self):
        return (1, 1)

    def set_backend(self, backend):
        super().set_backend(backend)
        self.observable.backend = backend


@dataclass
class State(QuantumDecoding):

    def __call__(self, x: Circuit) -> ndarray:
        state = super().__call__(x).state()
        return self.backend.np.vstack(
            (self.backend.np.real(state), self.backend.np.imag(state))
        )

    @property
    def output_shape(self):
        return (2, 2**self.nqubits)


@dataclass
class Samples(QuantumDecoding):

    def __post_init__(self):
        super().__post_init__()
        self.analytic = False

    def forward(self, x: Circuit) -> ndarray:
        return self.backend.cast(super().__call__(x).samples(), self.backend.precision)

    @property
    def output_shape(self):
        return (self.nshots, len(self.qubits))
