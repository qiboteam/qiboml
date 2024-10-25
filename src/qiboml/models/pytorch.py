"""Torch interface to qiboml layers"""

from dataclasses import dataclass

import torch
from qibo import Circuit
from qibo.backends import Backend

from qiboml.models.decoding import QuantumDecoding
from qiboml.models.encoding import QuantumEncoding


@dataclass(eq=False)
class QuantumModel(torch.nn.Module):

    encoding: QuantumEncoding
    circuit: Circuit
    decoding: QuantumDecoding
    differentiation_rule: callable = None

    def __post_init__(
        self,
    ):
        super().__init__()

        params = [p for param in self.circuit.get_parameters() for p in param]
        params = torch.as_tensor(self.backend.to_numpy(params)).ravel()
        params.requires_grad = True
        self.circuit_parameters = torch.nn.Parameter(params)

    def forward(self, x: torch.Tensor):
        if (
            self.backend.name != "pytorch"
            or self.differentiation_rule is not None
            or not self.decoding.analytic
        ):
            x = QuantumModelAutoGrad.apply(
                x,
                self.encoding,
                self.circuit,
                self.decoding,
                self.backend,
                self.differentiation_rule,
                *list(self.parameters())[0],
            )
        else:
            self.circuit.set_parameters(list(self.parameters())[0])
            x = self.encoding(x) + self.circuit
            x = self.decoding(x)

        return x

    @property
    def nqubits(
        self,
    ) -> int:
        return self.encoding.nqubits

    @property
    def backend(
        self,
    ) -> Backend:
        return self.decoding.backend

    @property
    def output_shape(self):
        return self.decoding.output_shape


class QuantumModelAutoGrad(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        encoding: QuantumEncoding,
        circuit: Circuit,
        decoding: QuantumDecoding,
        backend,
        differentiation_rule,
        *parameters: list[torch.nn.Parameter],
    ):
        ctx.save_for_backward(x, *parameters)
        ctx.encoding = encoding
        ctx.circuit = circuit
        ctx.decoding = decoding
        ctx.backend = backend
        ctx.differentiation_rule = differentiation_rule
        x_clone = x.clone().detach().cpu().numpy()
        x_clone = backend.cast(x_clone, dtype=x_clone.dtype)
        params = [
            backend.cast(par.clone().detach().cpu().numpy(), dtype=backend.precision)
            for par in parameters
        ]
        x_clone = encoding(x_clone) + circuit
        x_clone.set_parameters(params)
        x_clone = decoding(x_clone)
        x_clone = torch.as_tensor(backend.to_numpy(x_clone).tolist())
        return x_clone

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x, *parameters = ctx.saved_tensors
        x_clone = x.clone().detach().cpu().numpy()
        x_clone = ctx.backend.cast(x_clone, dtype=x_clone.dtype)
        params = [
            ctx.backend.cast(
                par.clone().detach().cpu().numpy(), dtype=ctx.backend.precision
            )
            for par in parameters
        ]
        grad_input, *gradients = (
            torch.as_tensor(ctx.backend.to_numpy(grad).tolist())
            for grad in ctx.differentiation_rule.evaluate(
                x_clone, ctx.encoding, ctx.circuit, ctx.decoding, ctx.backend, *params
            )
        )
        return (
            grad_output @ grad_input,
            None,
            None,
            None,
            None,
            None,
            *(torch.vstack(gradients).view((-1,) + grad_output.shape) @ grad_output.T),
        )
