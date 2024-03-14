"""Strategies for data encoding into a Parametrized Quantum Circuit."""

from qibo import Circuit


class EncodingCircuit:
    """
    An encoding circuit is a quantum circuit with a data encoding strategy.

    Args:
        circuit (Circuit): a Qibo circuit.
        encoding_strategy (callable): a callable function which defines the encoding
            strategy of the data inside the circuit.
    """

    def __init__(self, circuit: Circuit, encoding_strategy: callable):
        self.circuit = circuit
        self.encoding_strategy = encoding_strategy

    def inject_data(self, data):
        """Encode the data into ``circuit`` according to the chosen encoding strategy."""
        return self.encoding_strategy(self.circuit, data)
