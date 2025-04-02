"""Torch interface to qiboml layers"""

from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
import torch
from qibo import Circuit
from qibo.backends import Backend
from qibo.ui import plot_circuit

from qiboml.interfaces import utils
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
        circuit_structure (Union[List[QuantumEncoding, Circuit], Circuit]):
            a list of Qibo circuits and Qiboml encoding layers, which defines
            the complete structure of the model. The whole circuit will be mounted
            by sequentially add the elements of the given list. It is also possible
            to pass a single circuit, in the case a sequential structure is not needed.
        decoding (QuantumDecoding): the decoding layer.
        differentiation (Differentiation, optional): the differentiation engine,
            if not provided a default one will be picked following what described in the :ref:`docs <_differentiation_engine>`.
    """

    circuit_structure: Union[Circuit, List[Union[Circuit, QuantumEncoding]]]
    decoding: QuantumDecoding
    differentiation: Optional[Differentiation] = None

    def __post_init__(
        self,
    ):
        super().__init__()

        # The following code works with list of objects
        if isinstance(self.circuit_structure, Circuit):
            self.circuit_structure = [self.circuit_structure]

        params = utils.get_params_from_circuit_structure(self.circuit_structure)

        params = torch.as_tensor(self.backend.to_numpy(x=params)).ravel()
        params.requires_grad = True
        self.circuit_parameters = torch.nn.Parameter(params)

        if self.differentiation is None:
            self.differentiation = utils.get_default_differentiation(
                decoding=self.decoding,
                instructions=DEFAULT_DIFFERENTIATION,
            )

    def forward(self, x: Optional[torch.Tensor] = None):
        """
        Perform one forward pass of the model: encode the classical data
        in a quantum circuit, executes it and decodes it.

        Args:
            x (Optional[torch.tensor]): the input data, if required. Default is None.
        Returns:
            (torch.tensor): the computed outputs.
        """
        if self.differentiation is None:
            circuit = utils.circuit_from_structure(
                circuit_structure=self.circuit_structure,
                x=x,
            )

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

        encoding_layer = next(
            (
                circ
                for circ in self.circuit_structure
                if isinstance(circ, QuantumEncoding)
            ),
            None,
        )
        dummy_data = (
            self.decoding.backend.cast(np.zeros(len(encoding_layer.qubits)))
            if encoding_layer is not None
            else None
        )
        circuit = utils.circuit_from_structure(self.circuit_structure, dummy_data)
        if plt_drawing:
            _, fig = plot_circuit(circuit, **plt_kwargs)
            return fig
        else:
            circuit.draw()
            return str(circuit)


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
        # Save the flag and other context
        ctx.save_for_backward(x, *parameters)
        ctx.circuit_structure = circuit_structure
        ctx.decoding = decoding
        ctx.backend = backend
        ctx.differentiation = differentiation

        # Checking whether input data require grad
        x_requires_grad = x.requires_grad

        # Process the input
        x_clone = x.clone().detach().cpu().numpy()
        x_clone = backend.cast(x_clone, dtype=x_clone.dtype)
        params = [
            backend.cast(par.clone().detach().cpu().numpy(), dtype=x_clone.dtype)
            for par in parameters
        ]

        # Build the temporary circuit from the circuit structure.
        ctx.differentiable_encodings = x_requires_grad
        circuit = Circuit(decoding.nqubits)
        for circ in circuit_structure:
            if isinstance(circ, QuantumEncoding):
                circuit += circ(x_clone)
                # Record if any encoding is differentiable.
                if not circ.differentiable:
                    ctx.differentiable_encodings = False
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
            ctx.backend.cast(par.clone().detach().cpu().numpy(), dtype=x_clone.dtype)
            for par in parameters
        ]
        wrt_inputs = ctx.differentiable_encodings
        grad_input, *gradients = (
            torch.as_tensor(
                ctx.backend.to_numpy(grad).tolist(), dtype=x.dtype, device=x.device
            )
            for grad in ctx.differentiation.evaluate(
                x_clone,
                ctx.circuit_structure,
                ctx.decoding,
                ctx.backend,
                *params,
                wrt_inputs=wrt_inputs,
            )
        )
        gradients = torch.vstack(gradients).view((-1,) + grad_output.shape)
        left_indices = tuple(range(len(gradients.shape)))
        right_indices = left_indices[::-1][: len(gradients.shape) - 2] + (
            len(left_indices),
        )
        # TODO: grad_output.mT when we move to batching
        gradients = torch.einsum(gradients, left_indices, grad_output.T, right_indices)
        return (
            grad_output @ grad_input,
            None,
            None,
            None,
            None,
            *gradients,
        )
