import torch

import qiboml.models.encoding_decoding as encodings
from qiboml.models.abstract import QuantumCircuitLayer

for name, cls in encodings.__dict__.items():
    if isinstance(cls, type):
        pass


class TorchWrapper(torch.nn.Module):

    def __init__(self, model: QuantumCircuitLayer) -> None:
        super().__init__()
        self.model = model

    def forward(self, x, **kwargs):
        return self.model.forward(x, **kwargs)

    def backward(self, input_grad, **kwargs):
        return self.model.backward(input_grad, **kwargs)
