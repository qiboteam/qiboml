"""Some standard encoding and decoding layers"""

from qiboml.models.abstract import QuantumCircuitLayer


class QuantumEncodingLayer(QuantumCircuitLayer):
    pass


class PhaseEncodingLayer(QuantumEncodingLayer):
    pass


class AmplitudeEncodingLayer(QuantumEncodingLayer):
    pass


"""
   .
   .
   .
   .
"""


class QuantumDecodingLayer(QuantumCircuitLayer):
    pass


"""
   .
   .
   .
   .
"""
