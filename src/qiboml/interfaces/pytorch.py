"""Torch interface to qiboml layers"""

from dataclasses import dataclass
from typing import List, Optional, Union

import torch
from qibo import Circuit
from qibo.backends import Backend
from qibo.ui import plot_circuit

from qiboml.models.decoding import QuantumDecoding
from qiboml.models.encoding import QuantumEncoding
from qiboml.operations.differentiation import PSR, Differentiation, Jax

DEFAULT_DIFFERENTIATION = {
    "qiboml-pytorch": None,
    "qiboml-tensorflow": Jax,
    "qiboml-jax": Jax,
    "numpy": Jax,
}


@dataclass(eq=False)
class QuantumModel(torch.nn.Module):
    """
    The pytorch interface to qiboml models.

    Args:
        encoding (QuantumEncoding): the encoding layer.
        circuit (Circuit): the trainable circuit.
        decoding (QuantumDecoding): the decoding layer.
        differentiation (Differentiation, optional): the differentiation engine,
            if not provided a default one will be picked following what described in the :ref:`docs <_differentiation_engine>`.
    """

    circuit_structure: List[Union[QuantumEncoding, Circuit]]
    decoding: QuantumDecoding
    differentiation: Optional[Differentiation] = None

    def __post_init__(
        self,
    ):
        super().__init__()

        params = []
        for circ in self.circuit_structure:
            if not isinstance(circ, QuantumEncoding):
                params.extend([p for param in circ.get_parameters() for p in param])

        params = torch.as_tensor(self.backend.to_numpy(x=params)).ravel()
        params.requires_grad = True
        self.circuit_parameters = torch.nn.Parameter(params)

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
        """
        Perform one forward pass of the model: encode the classical data
        in a quantum circuit, executes it and decodes it.

        Args:
            x (torch.tensor): the input data.
        Returns:
            (torch.tensor): the computed outputs.
        """
        if self.differentiation is None:

            circuit = Circuit(self.nqubits)
            for circ in self.circuit_structure:
                if isinstance(circ, QuantumEncoding):
                    circuit += circ(x)
                else:
                    circuit += circ

            circuit.set_parameters(list(self.parameters())[0])
            x = self.decoding(circuit)
        else:
            x = QuantumModelAutoGrad.apply(
                x,
                self.circuit_structure,
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
        """
        The total number of qubits of the model.

        Returns:
            (int): the total number of qubits.
        """
        return self.decoding.nqubits

    @property
    def backend(
        self,
    ) -> Backend:
        """
        The execution backend of the model, which is inherited by the decoder.

        Returns:
            (Backend): the backend.
        """
        return self.decoding.backend

    @property
    def output_shape(self):
        """
        The shape of the output tensor produced by the model, which is
        defined by the decoder.

        Returns:
            (tuple(int)): the shape.
        """
        return self.decoding.output_shape

    def draw(self, plt_drawing=True, **plt_kwargs):
        """
        Draw the full circuit structure.

        Args:
            plt_drawing (bool): if True, the `qibo.ui.plot_circuit` function is used.
                If False, the default `circuit.draw` method is used.
            plt_kwargs (dict): extra arguments which can be set to customize the
                `qibo.ui.plot_circuit` function.
        """
        circuit = Circuit(self.nqubits)
        for circ in self.circuit_structure:
            if isinstance(circ, QuantumEncoding):
                circuit += circ.circuit
            else:
                circuit += circ

        if plt_drawing:
            plot_circuit(circuit, **plt_kwargs)
        else:
            circuit.draw()


class QuantumModelAutoGrad(torch.autograd.Function):
    """
    Custom Autograd to enable the autodifferentiation of the QuantumModel for
    non-pytorch backends.
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        circuit_structure: List[Union[QuantumEncoding, Circuit]],
        decoding: QuantumDecoding,
        backend,
        differentiation,
        *parameters: List[torch.nn.Parameter],
    ):
        ctx.save_for_backward(x, *parameters)
        ctx.circuit_structure = circuit_structure
        ctx.decoding = decoding
        ctx.backend = backend
        ctx.differentiation = differentiation

        x_clone = x.clone().detach().cpu().numpy()
        x_clone = backend.cast(x_clone, dtype=x_clone.dtype)
        params = [
            backend.cast(par.clone().detach().cpu().numpy(), dtype=backend.precision)
            for par in parameters
        ]

        # Build the temporary circuit based on the number of layers
        circuit = Circuit(decoding.nqubits)
        for circ in circuit_structure:
            if isinstance(circ, QuantumEncoding):
                circuit += circ(x_clone)
            else:
                circuit += circ

        circuit.set_parameters(params)
        x_clone = decoding(circuit)
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
        wrt_inputs = not x.is_leaf and ctx.encoding.differentiable
        # Evaluate the gradients using the differentiation engine
        grad_vals = ctx.differentiation.evaluate(
            x_clone,
            ctx.circuit_structure,
            ctx.decoding,
            ctx.backend,
            *params,
            wrt_inputs=wrt_inputs,
        )
        # Convert each gradient to a torch tenso
        grad_tensors = [
            torch.as_tensor(ctx.backend.to_numpy(g).tolist()) for g in grad_vals
        ]
        grad_input, *gradients = grad_tensors

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
            *gradients,
        )
