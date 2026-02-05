"""Torch interface to qiboml layers"""

from dataclasses import dataclass
from functools import reduce
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import torch
from qibo import Circuit
from qibo.backends import Backend
from qibo.config import raise_error

from qiboml.backends.pytorch import PyTorchBackend
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
    def jacfwd(f: Callable, argnums: Union[int, Tuple[int]]) -> Callable:
        return torch.func.jacfwd(f, argnums)

    @staticmethod
    def jacrev(f: Callable, argnums: Union[int, Tuple[int]]) -> Callable:
        return torch.func.jacrev(f, argnums)

    def identity(
        self, dim: int, dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        return torch.eye(dim, dtype=dtype, device=device)

    def zeros(
        self, shape: Union[int, Tuple[int]], dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        return torch.zeros(shape, dtype=dtype, device=device)

    def requires_gradient(self, x: torch.Tensor) -> bool:
        return not x.is_leaf or x.requires_grad


@dataclass(eq=False)
class QuantumModel(torch.nn.Module):
    """
    The pytorch interface to qiboml models.

    Args:
        circuit_structure (Union[List[QuantumEncoding, Circuit, Callable], Circuit]):
            a list of Qibo circuits and Qiboml encoding layers, which defines
            the complete structure of the model. The whole circuit will be mounted
            by sequentially stacking the elements of the given list. It is also possible
            to pass a single circuit, in the case a sequential structure is not needed.
        decoding (QuantumDecoding): the decoding layer.
        parameters_initialization (Union[keras.initializers.Initializer, np.ndarray]]): if an initialiser is provided it will be used
        either as the parameters or to sample the parameters of the model.
        differentiation (Differentiation, optional): the differentiation engine,
            if not provided a default one will be picked following what described in
            the :ref:`docs <_differentiation_engine>`.
        circuit_tracer (CircuitTracer, optional): tracer used to build the circuit
        and trace the operations performed upon construction. Defaults to ``TorchCircuitTracer``.
    """

    circuit_structure: Union[Circuit, List[Union[Circuit, QuantumEncoding, Callable]]]
    decoding: QuantumDecoding
    parameters_initialization: Optional[Union[np.ndarray, callable]] = None
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
        params = torch.as_tensor(self.backend.to_numpy(x=params)).ravel()

        if self.parameters_initialization is not None:
            if callable(self.parameters_initialization):
                params = torch.empty(
                    params.shape, dtype=params.dtype, requires_grad=True
                )
                params = self.parameters_initialization(params)
            elif isinstance(self.parameters_initialization, np.ndarray | torch.Tensor):
                if self.parameters_initialization.shape != params.shape:
                    raise_error(
                        ValueError,
                        f"Shape not valid for `parameters_initialization`. The shape should be {params.shape}.",
                    )
                params = torch.as_tensor(self.parameters_initialization).ravel()
            else:
                raise_error(
                    ValueError,
                    "`parameters_initialization` should be a `np.ndarray` or `torch.nn.init`.",
                )
        params.requires_grad = True
        self.circuit_parameters = torch.nn.Parameter(params)

        if self.circuit_tracer is None:
            self.circuit_tracer = TorchCircuitTracer
        self.circuit_tracer = self.circuit_tracer(self.circuit_structure)

        if self.differentiation is None:
            if (
                issubclass(type(self.backend), PyTorchBackend)
                and self.decoding.analytic
            ):
                self.differentiation = None
            else:
                self.differentiation = utils.get_default_differentiation(
                    decoding=self.decoding,
                    instructions=DEFAULT_DIFFERENTIATION,
                )()
        elif isinstance(self.differentiation, type):
            self.differentiation = self.differentiation()

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
            if not self.differentiation._is_built:
                self.differentiation.build(
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
            self,
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
        # convert the parameters to backend native arrays
        dtype = getattr(decoding.backend.np, str(parameters.dtype).split(".")[-1])
        angles = decoding.backend.cast(
            angles.detach().cpu().clone().numpy(), dtype=dtype
        )
        for g, p in zip(differentiation.circuit.parametrized_gates, angles):
            g.parameters = p
        circuit = differentiation.circuit
        # circuit.set_parameters(params)

        wrt_inputs = jacobian_wrt_inputs is not None
        if not wrt_inputs:
            angles = decoding.backend.cast(
                [par for params in circuit.get_parameters() for par in params],
                dtype=dtype,
            )

        # Save the context
        ctx.save_for_backward(jacobian_wrt_inputs, jacobian)
        ctx.circuit = circuit
        ctx.angles = angles
        ctx.differentiation = differentiation
        ctx.input_to_gate_map = input_to_gate_map
        ctx.dtype = dtype
        ctx.wrt_inputs = jacobian_wrt_inputs is not None

        x_clone = decoding(circuit)
        x_clone = torch.as_tensor(
            decoding.backend.to_numpy(x_clone).tolist(),
            dtype=parameters.dtype,
            device=parameters.device,
        )
        return x_clone

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        jacobian_wrt_inputs, jacobian = ctx.saved_tensors
        backend = ctx.differentiation.decoding.backend
        params = ctx.angles
        # get the jacobian of the output wrt each angle of the circuit
        # (i.e. each rotation gate)
        jacobian_wrt_angles = torch.as_tensor(
            backend.to_numpy(
                ctx.differentiation.evaluate(params, wrt_inputs=ctx.wrt_inputs)
            ),
            dtype=jacobian.dtype,
            device=jacobian.device,
        )
        out_shape = ctx.differentiation.decoding.output_shape
        # contraction to combine jacobians wrt inputs/parameters with those
        # wrt the circuit angles
        contraction = ((0, 1), (0,) + tuple(range(2, len(out_shape) + 2)))
        # contraction to combine with the gradients coming from outside
        right_indices = tuple(range(1, len(grad_output.shape) + 1))
        left_indices = (0,) + right_indices

        if jacobian_wrt_inputs is not None:
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
            jacobian_wrt_angles = torch.vstack(
                [
                    row
                    for i, row in enumerate(jacobian_wrt_angles)
                    if i not in indices_to_discard
                ]
            ).reshape(-1, *out_shape)
            # combine the jacobians wrt inputs with those
            # wrt the circuit angles
            grad_input = torch.einsum(
                jacobian_wrt_inputs,
                contraction[0],
                jacobian_wrt_encoding_angles,
                contraction[1],
            )
            grad_input = torch.einsum(
                grad_input, left_indices, grad_output, right_indices
            )
        else:
            grad_input = None
        # combine the jacobians wrt parameters with those
        # wrt the circuit angles
        gradient = torch.einsum(
            jacobian, contraction[0], jacobian_wrt_angles, contraction[1]
        )
        # combine with the gradients coming from outside
        gradient = torch.einsum(gradient, left_indices, grad_output, right_indices)
        return (
            grad_input,
            None,
            None,
            None,
            *gradient,
        )
