import pytest
from qibo import Circuit

from qiboml.interfaces.utils import circuit_from_structure


def test_circuit_from_structure():
    c = [Circuit(2), 1]
    with pytest.raises(RuntimeError):
        circuit_from_structure(c, params=[])
