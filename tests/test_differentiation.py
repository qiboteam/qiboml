import pytest
from qibo import Circuit, gates

from qiboml.models.decoding import Expectation, Probabilities
from qiboml.operations.differentiation import PSR
from qiboml.operations.differentiation_jax import Jax


def test_PSR_decoder_error():
    decoder = Probabilities(2)
    circuit = Circuit(2)
    with pytest.raises(RuntimeError):
        PSR(circuit, decoder)


def test_Jax_setters():
    decoder = Probabilities(1)
    circuit = Circuit(1)
    circuit.add([gates.RX(0, theta=0.0) for _ in range(3)])
    diff = Jax(circuit, decoder)
    diff.circuit = Circuit(1)
    assert len(diff.circuit.queue) == 0
    diff.decoding = Expectation(1)
    assert isinstance(diff.decoding, Expectation)
