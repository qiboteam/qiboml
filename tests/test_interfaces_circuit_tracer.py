import numpy as np
import pytest
from qibo import Circuit

from qiboml.models.encoding import PhaseEncoding


def get_tracer(frontend):
    if frontend.__name__ == "qiboml.interfaces.pytorch":
        return frontend.TorchCircuitTracer
    else:
        return frontend.KerasCircuitTracer


def test_circuit_tracer_no_encodings(frontend):
    tracer = get_tracer(frontend)

    def f(a):
        return Circuit(1)

    circuit_structure = [Circuit(1), f]
    tracer = tracer(circuit_structure)
    assert not tracer.is_encoding_differentiable


def test_circuit_tracer_no_input_error(frontend):
    tracer = get_tracer(frontend)
    circuit_structure = [Circuit(1), PhaseEncoding(1)]
    tracer = tracer(circuit_structure)
    with pytest.raises(ValueError):
        tracer(
            params=[],
        )
