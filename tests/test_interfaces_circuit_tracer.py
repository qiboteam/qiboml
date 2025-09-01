import numpy as np
import pytest
from qibo import Circuit

from qiboml.interfaces.keras import KerasCircuitTracer
from qiboml.interfaces.pytorch import TorchCircuitTracer
from qiboml.models.encoding import PhaseEncoding

TRACERS = [TorchCircuitTracer, KerasCircuitTracer]


@pytest.mark.parametrize("tracer", TRACERS)
def test_circuit_tracer_no_encodings(tracer):

    def f(a):
        return Circuit(1)

    circuit_structure = [Circuit(1), f]
    tracer = tracer(circuit_structure)
    assert not tracer.is_encoding_differentiable


@pytest.mark.parametrize("tracer", TRACERS)
def test_circuit_tracer_no_input_error(tracer):
    circuit_structure = [Circuit(1), PhaseEncoding(1)]
    tracer = tracer(circuit_structure)
    with pytest.raises(ValueError):
        tracer(
            params=[],
        )
