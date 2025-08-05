"""Torch interface to qiboml layers"""

from dataclasses import dataclass
from functools import reduce
from typing import Callable, List, Optional, Union

import numpy as np
import torch
from qibo import Circuit
from qibo.backends import Backend
from qibo.config import raise_error

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
            by sequentially stacking the elements of the given list. It is also possible
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
        if isinstance(self.circuit_structure, Circuit):
            self.circuit_structure = [self.circuit_structure]

        params = utils.get_params_from_circuit_structure(
            self.circuit_structure,
        )
        # params = torch.as_tensor(self.backend.to_numpy(x=params)).ravel()
        params = torch.as_tensor(params).ravel()
        params.requires_grad = True
        self.circuit_parameters = torch.nn.Parameter(params)

        if self.differentiation is None:
            self.differentiation = utils.get_default_differentiation(
                decoding=self.decoding,
                instructions=DEFAULT_DIFFERENTIATION,
            )

        if any(
            isinstance(circuit, Callable)
            and not isinstance(circuit, QuantumEncoding | Circuit)
            for circuit in self.circuit_structure
        ) and (
            isinstance(self.differentiation, PSR)
        ):  # pragma: no cover
            raise_error(
                NotImplementedError, "Equivariant circuits not working with PSR yet."
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
                params=list(self.parameters())[0],
                engine=torch,
                x=x,
            )
            x = self.decoding(circuit)
        else:
            x = QuantumModelAutoGrad_new.apply(
                x,
                self.circuit_structure,
                self.decoding,
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
        circuit_structure: List[Union[QuantumEncoding, Circuit, Callable]],
        decoding: QuantumDecoding,
        backend,
        differentiation,
        *parameters: List[torch.nn.Parameter],
    ):
        # Save the context
        ctx.save_for_backward(x, *parameters)
        ctx.circuit_structure = circuit_structure
        ctx.decoding = decoding
        ctx.backend = backend
        ctx.differentiation = differentiation
        dtype = getattr(backend.np, str(parameters[0].dtype).split(".")[-1])
        ctx.dtype = dtype

        if x is not None:
            # Cloning, detaching and converting to backend arrays
            x_clone = x.clone().detach().cpu().numpy()
            x_clone = backend.cast(x_clone, dtype=x_clone.dtype)
        else:
            x_clone = x
        params = [
            backend.cast(par.clone().detach().cpu().numpy(), dtype=dtype)
            for par in parameters
        ]

        # all the encodings need to be differentiatble
        # TODO: open to debate
        ctx.differentiable_encodings = all(
            enc.differentiable
            for enc in circuit_structure
            if isinstance(enc, QuantumEncoding)
        )
        circuit = utils.circuit_from_structure(
            circuit_structure, x_clone, params, backend=backend
        )
        x_clone = decoding(circuit)
        x_clone = torch.as_tensor(
            backend.to_numpy(x_clone).tolist(),
            dtype=parameters[0].dtype,
            device=parameters[0].device,
        )
        return x_clone

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x, *parameters = ctx.saved_tensors
        if x is not None:
            x_clone = x.clone().detach().cpu().numpy()
            x_clone = ctx.backend.cast(x_clone, dtype=x_clone.dtype)
            wrt_inputs = (
                not x.is_leaf or x.requires_grad
            ) and ctx.differentiable_encodings
        else:
            x_clone = None
            wrt_inputs = False
        params = [
            ctx.backend.cast(par.clone().detach().cpu().numpy(), dtype=ctx.dtype)
            for par in parameters
        ]
        grad_input, *gradients = (
            torch.as_tensor(
                ctx.backend.to_numpy(grad).tolist(),
                dtype=parameters[0].dtype,
                device=parameters[0].device,
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
        grad_input = grad_output @ grad_input if x is not None else None
        return (
            grad_input,
            None,
            None,
            None,
            None,
            *gradients,
        )


class QuantumModelAutoGrad_new(torch.autograd.Function):
    """
    Custom Autograd to enable the autodifferentiation of the QuantumModel for
    non-pytorch backends.
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        circuit_structure: List[Union[QuantumEncoding, Circuit, Callable]],
        decoding: QuantumDecoding,
        differentiation,
        *parameters: List[torch.nn.Parameter],
    ):
        tracer = circuit_trace  # if torch.is_grad_enabled() else None
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
        circuit, jacobian_wrt_inputs, jacobian, input_to_gate_map = (
            utils.circuit_from_structure(
                circuit_structure, parameters, torch, x, tracer
            )
        )
        jacobian = jacobian.to(parameters.device)
        params = torch.stack(
            [
                par
                for params in circuit.get_parameters(include_not_trainable=True)
                for par in params
            ]
        )
        # Save the context
        ctx.save_for_backward(jacobian_wrt_inputs, jacobian, params)
        ctx.circuit = circuit
        ctx.differentiation = differentiation
        ctx.input_to_gate_map = input_to_gate_map
        dtype = getattr(decoding.backend.np, str(parameters.dtype).split(".")[-1])
        ctx.dtype = dtype
        # convert the parameters to backend native arrays
        # params = [
        #    decoding.backend.cast(par.clone().detach().cpu().numpy(), dtype=dtype)
        #    for par in params
        # ]
        params = decoding.backend.cast(
            params.clone().detach().cpu().numpy(), dtype=dtype
        )
        for g, p in zip(circuit.parametrized_gates, params):
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
        # params = [
        #    backend.cast(par.clone().detach().cpu().numpy(), dtype=ctx.dtype)
        #    for par in parameters
        # ]
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
        right_indices = left_indices[
            1:
        ]  # left_indices[::-1][: len(gradients.shape) - 2] + (
        # len(left_indices),
        # )
        # TODO: grad_output.mT when we move to batching
        gradients = torch.einsum(gradients, left_indices, grad_output, right_indices)
        grad_input = torch.einsum(
            grad_input, left_indices, grad_output, right_indices
        )  # grad_output @ grad_input
        return (
            grad_input,
            None,
            None,
            None,
            *gradients,
        )


def circuit_trace(f: Callable, params):

    circuit = None

    def build(x):
        nonlocal circuit
        is_encoding = isinstance(f, QuantumEncoding)
        circuit = f(x) if is_encoding else f(*x)
        # one parameter gates only
        return torch.vstack(
            [
                par[0]
                for par in circuit.get_parameters(include_not_trainable=is_encoding)
            ]
        )

    # fwd or bkwd mode?
    # we always assume the input is a 1-dim array, even for encodings
    # thus the jacobian is a matrix
    jac = torch.autograd.functional.jacobian(build, params).reshape(-1, len(params))
    par_map = {}
    for i, row in enumerate(jac):
        for j in torch.nonzero(row):
            j = int(j)
            if j in par_map:
                par_map[j] += (i,)
            else:
                par_map[j] = (i,)
    return jac, par_map, circuit
