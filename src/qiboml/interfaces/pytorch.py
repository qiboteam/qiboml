"""Torch interface to qiboml layers"""

from dataclasses import dataclass
from functools import reduce
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import torch
from qibo import Circuit
from qibo.backends import Backend
from qibo.config import raise_error

from qiboml.interfaces import utils
from qiboml.interfaces.circuit_tracer import CircuitTracer
from qiboml.models.decoding import QuantumDecoding
from qiboml.models.encoding import QuantumEncoding
from qiboml.operations.differentiation import PSR, Differentiation, Jax

DEFAULT_DIFFERENTIATION = {
    "qiboml-pytorch": None,
    "qiboml-tensorflow": Jax,
    "qiboml-jax": Jax,
    "numpy": Jax,
}


class TorchCircuitTracer(CircuitTracer):

    @property
    def engine(self):
        return torch

    @staticmethod
    def jacfwd(f: Callable, argnums: Union[int, Tuple[int]]):
        return torch.func.jacfwd(f, argnums)

    @staticmethod
    def jacrev(f: Callable, argnums: Union[int, Tuple[int]]):
        return torch.func.jacrev(f, argnums)

    def identity(
        self, dim: int, dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        return torch.eye(dim, dtype=dtype, device=device)

    def zeros(
        self, shape: Union[int, Tuple[int]], dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        return torch.zeros(shape, dtype=dtype, device=device)


@dataclass(eq=False)
class QuantumModel(torch.nn.Module):
    """
    The pytorch interface to qiboml models.

    Args:
        circuit_structure (Union[List[QuantumEncoding, Circuit], Circuit]):
            a list of Qibo circuits and Qiboml encoding layers, which defines
            the complete structure of the model. The whole circuit will be mounted
            by sequentially stacking the elements of the given list. It is also possible
            to pass a single circuit, in the case a sequential structure is not needed.
        decoding (QuantumDecoding): the decoding layer.
        differentiation (Differentiation, optional): the differentiation engine,
            if not provided a default one will be picked following what described in the :ref:`docs <_differentiation_engine>`.
    """

    circuit_structure: Union[Circuit, List[Union[Circuit, QuantumEncoding]]]
    decoding: QuantumDecoding
    differentiation: Optional[Differentiation] = None
    circuit_tracer: Optional[CircuitTracer] = None

    def __post_init__(
        self,
    ):
        super().__init__()
        if isinstance(self.circuit_structure, Circuit):
            self.circuit_structure = [self.circuit_structure]

        params = utils.get_params_from_circuit_structure(
            self.circuit_structure,
        )
        # params = torch.as_tensor(self.backend.to_numpy(x=params)).ravel()
        params = torch.as_tensor(params).ravel()
        params.requires_grad = True
        self.circuit_parameters = torch.nn.Parameter(params)

        if self.circuit_tracer is None:
            self.circuit_tracer = TorchCircuitTracer
        self.circuit_tracer = self.circuit_tracer(self.circuit_structure)

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
            circuit = self.circuit_tracer.build_circuit(
                params=list(self.parameters())[0],
                x=x,
            )
            x = self.decoding(circuit)
        else:
            if isinstance(self.differentiation, type):
                self.differentiation = self.differentiation(
                    self.circuit_tracer.build_circuit(list(self.parameters())[0], x=x),
                    self.decoding,
                )
            x = QuantumModelAutoGrad.apply(
                x,
                self.decoding,
                self.differentiation,
                self.circuit_tracer,
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

        fig = utils.draw_circuit(
            circuit_structure=self.circuit_structure,
            backend=self.decoding.backend,
            plt_drawing=plt_drawing,
            **plt_kwargs,
        )

        return fig


class QuantumModelAutoGrad(torch.autograd.Function):
    """
    Custom Autograd to enable the autodifferentiation of the QuantumModel for
    non-pytorch backends.
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        decoding: QuantumDecoding,
        differentiation,
        circuit_tracer: TorchCircuitTracer,
        *parameters: List[torch.nn.Parameter],
    ):
        """
        # all the encodings need to be differentiatble
        # TODO: open to debate
        ctx.differentiable_encodings = all(
            enc.differentiable
            for enc in circuit_structure
            if isinstance(enc, QuantumEncoding)
        )
        """
        parameters = torch.stack(parameters)
        # it would be maybe better to perform the tracing in the backward only
        # this way the jacobians are calculated only if the backward is called
        circuit, jacobian_wrt_inputs, jacobian, input_to_gate_map = circuit_tracer(
            parameters, x=x
        )
        angles = torch.stack(
            [
                par
                for params in circuit.get_parameters(include_not_trainable=True)
                for par in params
            ]
        )
        # Save the context
        ctx.save_for_backward(jacobian_wrt_inputs, jacobian, angles)
        ctx.circuit = circuit
        ctx.differentiation = differentiation
        ctx.input_to_gate_map = input_to_gate_map
        dtype = getattr(decoding.backend.np, str(parameters.dtype).split(".")[-1])
        ctx.dtype = dtype
        # convert the parameters to backend native arrays
        angles = decoding.backend.cast(
            angles.clone().detach().cpu().numpy(), dtype=dtype
        )
        for g, p in zip(circuit.parametrized_gates, angles):
            g.parameters = p
        # circuit.set_parameters(params)
        x_clone = decoding(circuit)
        x_clone = torch.as_tensor(
            decoding.backend.to_numpy(x_clone).tolist(),
            dtype=parameters.dtype,
            device=parameters.device,
        )
        return x_clone

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        jacobian_wrt_inputs, jacobian, parameters = ctx.saved_tensors
        backend = ctx.differentiation.decoding.backend
        params = backend.cast(
            parameters.clone().detach().cpu().numpy(), dtype=ctx.dtype
        )
        # get the jacobian of the output wrt each angle of the circuit
        # (i.e. each rotation gate)
        jacobian_wrt_angles = torch.as_tensor(
            backend.to_numpy(
                ctx.differentiation.evaluate(
                    params,
                )
            ),
            dtype=parameters[0].dtype,
            device=parameters[0].device,
        )

        # extract the rows corresponding to encoding gates
        # thus those element to be combined with the jacobian
        # wrt the inputs
        jacobian_wrt_encoding_angles = torch.vstack(
            [
                jacobian_wrt_angles[list(indices)]
                for indices in zip(*ctx.input_to_gate_map.values())
            ]
        )
        # discard the elements corresponding to encoding gates
        # to obtain only the part wrt the model's parameters
        indices_to_discard = reduce(tuple.__add__, ctx.input_to_gate_map.values())
        out_shape = ctx.differentiation.decoding.output_shape
        jacobian_wrt_angles = torch.vstack(
            [
                row
                for i, row in enumerate(jacobian_wrt_angles)
                if i not in indices_to_discard
            ]
        ).reshape(-1, *out_shape)

        # combine the jacobians wrt parameters/inputs with those
        # wrt the circuit angles
        contraction = ((0, 1), (0,) + tuple(range(2, len(out_shape) + 2)))
        gradients = torch.einsum(
            jacobian, contraction[0], jacobian_wrt_angles, contraction[1]
        )
        grad_input = torch.einsum(
            jacobian_wrt_inputs,
            contraction[0],
            jacobian_wrt_encoding_angles,
            contraction[1],
        )

        # combine with the gradients coming from outside
        left_indices = tuple(range(len(gradients.shape)))
        right_indices = left_indices[1:]
        gradients = torch.einsum(gradients, left_indices, grad_output, right_indices)
        grad_input = torch.einsum(grad_input, left_indices, grad_output, right_indices)
        return (
            grad_input,
            None,
            None,
            None,
            *gradients,
        )
