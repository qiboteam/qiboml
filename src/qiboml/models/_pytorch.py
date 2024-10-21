from dataclasses import dataclass

import torch
from qibo import Circuit

from qiboml.models.decoding import DecodingCircuit
from qiboml.models.encoding import EncodingCircuit


@dataclass(eq=False)
class QuantumModel(torch.nn.Module):

    encoding: EncodingCircuit
    training: Circuit
    decoding: DecodingCircuit

    def __post_init__(self):

        for layer in (self.encoding, self.training, self.decoding):
            setattr(
                self,
                f"{layer.__class__.__name__}",
                torch.nn.Parameter(params.squeeze()),
            )
