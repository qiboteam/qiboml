"""Torch interface to qiboml layers"""

from dataclasses import dataclass

import numpy as np
import torch
from qibo import Circuit
from qibo.backends import Backend
from qibo.config import raise_error

from qiboml.models.decoding import QuantumDecoding
from qiboml.models.encoding import QuantumEncoding
from qiboml.operations.differentiation import PSR, Differentiation, Jax

DEFAULT_DIFFERENTIATION = {
    "qiboml-pytorch": None,
    "qiboml-tensorflow": Jax,
    "qiboml-jax": Jax,
}


@dataclass(eq=False)
class QuantumModel(torch.nn.Module):

    encoding: QuantumEncoding
    circuit: Circuit
    decoding: QuantumDecoding
    differentiation: Differentiation = None

    def __post_init__(
        self,
    ):
        super().__init__()

        params = [p for param in self.circuit.get_parameters() for p in param]
        params = torch.as_tensor(self.backend.to_numpy(x=params)).ravel()
        params.requires_grad = True
        self.circuit_parameters = torch.nn.Parameter(params)
        # This is used to know if the non-differentiable gates have been marked
        # so to reduce the computational cost of hardware-compatible differentiation
        self._differentiability_checked = False

        backend_string = (
            f"{self.decoding.backend.name}-{self.decoding.backend.platform}"
            if self.decoding.backend.platform is not None
            else self.decoding.backend.name
        )

        if self.differentiation is None:
            if not self.decoding.analytic:
                self.differentiation = PSR()
            else:
                if backend_string in DEFAULT_DIFFERENTIATION.keys():
                    diff = DEFAULT_DIFFERENTIATION[backend_string]
                    self.differentiation = diff() if diff is not None else None
                else:
                    self.differentiation = PSR()

    def forward(self, x: torch.Tensor):
        if self.differentiation is None:
            self.circuit.set_parameters(list(self.parameters())[0])
            x = self.encoding(x) + self.circuit
            x = self.decoding(x)
        else:
            # Also for the first iteration, marking which gates are differentiable
            if not self._differentiability_checked:
                self._mark_differentiable_angles(
                    circuit=self.encoding(x) + self.circuit,
                    differentiate_wrt_data=False,  # TODO: expose this feature if we think it's useful
                )
                # Inform the model the check is done
                self._differentiability_checked = True

            x = QuantumModelAutoGrad.apply(
                x,
                self.encoding,
                self.circuit,
                self.decoding,
                self.backend,
                self.differentiation,
                *list(self.parameters())[0],
            )
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

    def _mark_differentiable_angles(self, circuit, differentiate_wrt_data=False):
        """
        Check circuit's parameters and identify which, among them, are not differentiable.
        This will be useful to reduce the computational cost of the parameter shift
        rule, by avoiding the computation of gradients w.r.t. input data.
        The distinction is made by setting the gates containing non differentiable
        angles as `trainable=False` within Qibo.

        Args:
            circuit (qibo.Circuit): qibo circuit to be checked.
            differentiate_wrt_data (bool): if True, gradient w.r.t. input data
                are computed as well. Default to `False`.
        """

        if not differentiate_wrt_data:
            for gate in circuit.parametrized_gates:
                if any(param.requires_grad == True for param in gate.parameters):
                    gate.trainable = True
                else:
                    gate.trainable = False


class QuantumModelAutoGrad(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        encoding: QuantumEncoding,
        circuit: Circuit,
        decoding: QuantumDecoding,
        backend,
        differentiation,
        *parameters: list[torch.nn.Parameter],
    ):
        ctx.save_for_backward(x, *parameters)
        ctx.encoding = encoding
        ctx.circuit = circuit
        ctx.decoding = decoding
        ctx.backend = backend
        ctx.differentiation = differentiation
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
            for grad in ctx.differentiation.evaluate(
                x_clone, ctx.encoding, ctx.circuit, ctx.decoding, ctx.backend, *params
            )
        )
        gradients = torch.vstack(gradients).view((-1,) + grad_output.shape)
        left_indices = tuple(range(len(gradients.shape)))
        right_indices = left_indices[::-1][: len(gradients.shape) - 2] + (
            len(left_indices),
        )
        gradients = torch.einsum(gradients, left_indices, grad_output.T, right_indices)
        return (
            grad_output @ grad_input,
            None,
            None,
            None,
            None,
            None,
            *gradients,
        )
