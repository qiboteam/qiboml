"""Keras interface to qiboml layers"""

# pylint: disable=no-member
# pylint: disable=unexpected-keyword-arg

import string
from dataclasses import dataclass
from functools import reduce
from logging import warning
from typing import Callable, List, Optional, Tuple, Union

import keras
import numpy as np
import tensorflow as tf  # pylint: disable=import-error
from qibo import Circuit
from qibo.backends import Backend
from qibo.config import raise_error

from qiboml.interfaces import utils
from qiboml.interfaces.circuit_tracer import CircuitTracer
from qiboml.models.decoding import QuantumDecoding
from qiboml.models.encoding import QuantumEncoding
from qiboml.operations.differentiation import PSR, Differentiation, Jax

DEFAULT_DIFFERENTIATION = {
    "qiboml-pytorch": Jax,
    "qiboml-tensorflow": None,
    "qiboml-jax": Jax,
    "numpy": Jax,
}


class KerasCircuitTracer(CircuitTracer):

    @property
    def engine(self):
        return keras.ops

    @staticmethod
    def jacrev(f: Callable, argnums: Union[int, Tuple[int]]) -> Callable:

        @tf.function
        def jac_functional(x):  # pragma: no cover
            with tf.GradientTape() as tape:
                tape.watch(x)
                y = f(x)
            return tape.jacobian(y, x)

        return jac_functional

    @staticmethod
    def jacfwd(f: Callable, argnums: Union[int, Tuple[int]]) -> Callable:
        warning(
            "No available implementation of forward differentiation in tensorflow, falling back to reverse mode differentiation."
        )
        return KerasCircuitTracer.jacrev(f, argnums)

    def nonzero(self, array: tf.Tensor) -> tf.Tensor:
        return self.engine.nonzero(array)[0]

    def identity(self, dim: int, dtype, device: str) -> tf.Tensor:
        with tf.device(device):
            eye = keras.ops.eye(dim, dtype=dtype)
        return eye

    def zeros(self, shape: Union[int, Tuple[int]], dtype, device: str) -> tf.Tensor:
        with tf.device(device):
            z = keras.ops.zeros(shape, dtype=dtype)
        return z

    def _build_parameters_map(self, jacobian):
        if tf.is_symbolic_tensor(jacobian):
            # just a placeholder map to continue symbolic computation
            return {0: (0,)}
        return super()._build_parameters_map(jacobian)

    def fill_jacobian(
        self,
        jacobian: tf.Tensor,
        row_span: Tuple[int, int],
        col_span: Tuple[int, int],
        values: tf.Tensor,
    ) -> tf.Tensor:
        return keras.ops.slice_update(jacobian, (row_span[0], col_span[0]), values)

    def requires_gradient(self, x: tf.Tensor) -> bool:
        if tf.is_symbolic_tensor(x):
            return hasattr(x, "op") and len(x.op.inputs) > 0
        return True


@dataclass(eq=False)
class QuantumModel(keras.Model):  # pylint: disable=no-member
    """
    The Keras interface to qiboml models.

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
            if not provided a default one will be picked following what described in the :ref:`docs <_differentiation_engine>`.
        circuit_tracer (CircuitTracer, optional): tracer used to build the circuit and trace the operations performed upon construction. Defaults to ``KerasCircuitTracer``.
    """

    circuit_structure: Union[Circuit, List[Union[Circuit, QuantumEncoding, Callable]]]
    decoding: QuantumDecoding
    parameters_initialization: Optional[
        Union[keras.initializers.Initializer, np.ndarray]
    ] = None
    differentiation: Optional[Differentiation] = None
    circuit_tracer: Optional[CircuitTracer] = None


    def __post_init__(self):
        super().__init__()

        if isinstance(self.circuit_structure, Circuit):
            self.circuit_structure = [self.circuit_structure]

        params = utils.get_params_from_circuit_structure(self.circuit_structure)
        params = keras.ops.cast(
            self.backend.to_numpy(params),
            "float64",
        )  # pylint: disable=no-member

        initializer = "zeros"
        if self.parameters_initialization is not None:
            if isinstance(self.parameters_initialization, keras.initializers.Initializer):
                initializer = self.parameters_initialization
            elif isinstance(self.parameters_initialization, np.ndarray | tf.Tensor):
                if self.parameters_initialization.shape != params.shape:
                    raise_error(
                        ValueError,
                        f"Shape not valid for `parameters_initialization`. The shape should be {params.shape}.",
                    )
                params = self.parameters_initialization
            else:
                raise_error(ValueError, "`parameters_initialization` should be a `np.ndarray` or `keras.initializers.Initializer`.")
        self.circuit_parameters = self.add_weight(
            shape=params.shape,
            initializer=initializer,
            trainable=True,

        )
        if not isinstance(self.parameters_initialization, keras.initializers.Initializer):
            self.set_weights([params])

        if self.circuit_tracer is None:
            self.circuit_tracer = KerasCircuitTracer
        self.circuit_tracer = self.circuit_tracer(self.circuit_structure)

        if self.differentiation is None:
            self.differentiation = utils.get_default_differentiation(
                decoding=self.decoding,
                instructions=DEFAULT_DIFFERENTIATION,
            )
        self.custom_gradient = None

    def compute_output_shape(self, input_shape):
        return self.decoding.output_shape

    def call(self, x: Optional[tf.Tensor] = None) -> tf.Tensor:
        if self.differentiation is None:
            # This `1 * self.circuit_parameters` is needed otherwise a TypeError is raised
            circuit = self.circuit_tracer.build_circuit(
                params=1 * self.circuit_parameters,
                x=x,
            )
            output = self.decoding(circuit)
            return output[None, :]
        if self.custom_gradient is None:
            self.differentiation = self.differentiation(
                self.circuit_tracer.build_circuit(1 * self.circuit_parameters, x=x),
                self.decoding,
            )
            self.custom_gradient = QuantumModelCustomGradient(
                self.decoding,
                self.differentiation,
                self.circuit_tracer,
            )
        if x is None:
            x = tf.constant([], dtype=np.float64)
        return self.custom_gradient.evaluate(x, 1 * self.circuit_parameters)

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

    
    @property
    def output_shape(
        self,
    ):
        return self.decoding.output_shape

    @property
    def nqubits(
        self,
    ) -> int:
        return self.decoding.nqubits

    @property
    def backend(
        self,
    ) -> Backend:
        return self.decoding.backend


@dataclass
class QuantumModelCustomGradient:

    decoding: QuantumDecoding
    differentiation: Differentiation
    circuit_tracer: CircuitTracer

    @property
    def backend(self):
        return self.decoding.backend

    @tf.custom_gradient
    def evaluate(self, x, params):
        if x.shape[0] == 0:
            x = None
        circuit, jacobian_wrt_inputs, jacobian, input_to_gate_map = self.circuit_tracer(
            params, x=x
        )
        wrt_inputs = jacobian_wrt_inputs is not None
        angles = keras.ops.stack(
            [
                par
                for params in circuit.get_parameters(include_not_trainable=True)
                for par in params
            ]
        )

        def forward(angles):
            angles = self.backend.cast(angles, dtype=self.backend.np.float64)
            for i, g in enumerate(self.differentiation.circuit.parametrized_gates):
                g.parameters = angles[i]
            circuit = self.differentiation.circuit
            y = self.decoding(circuit)
            y = self.backend.to_numpy(y)
            return y

        y = tf.numpy_function(func=forward, inp=[angles], Tout=tf.float64)
        y = keras.ops.reshape(y, self.decoding.output_shape)

        def jacobian_wrt_angles(angles):
            angles = self.backend.cast(angles, dtype=self.backend.np.float64)
            d_angles = self.differentiation.evaluate(
                angles,
                wrt_inputs=wrt_inputs,
            )
            d_angles = self.backend.to_numpy(
                self.backend.cast(d_angles, dtype=self.backend.np.float64)
            )
            return d_angles

        # Custom gradient
        def grad(dy):
            d_angles = tf.numpy_function(
                func=jacobian_wrt_angles, inp=[angles], Tout=[tf.float64]
            )
            # in symbolic execution d_angles is returned as a list containing just one
            # tensor for some reason... thus the added vstack
            d_angles = keras.ops.vstack(d_angles)
            out_shape = self.differentiation.decoding.output_shape
            # contraction to combine jacobians wrt inputs/parameters with those
            # wrt the circuit angles
            contraction = ((0, 1), (0,) + tuple(range(2, len(out_shape) + 2)))
            contraction = ",".join(
                "".join(string.ascii_letters[i] for i in indices)
                for indices in contraction
            )
            # contraction to combine with the gradients coming from outside
            rhs = "".join(
                string.ascii_letters[i] for i in tuple(range(1, len(dy.shape) + 1))
            )
            lhs = string.ascii_letters[0] + rhs

            if jacobian_wrt_inputs is not None:
                if tf.is_symbolic_tensor(d_angles):
                    d_angles = keras.ops.reshape(
                        d_angles, (angles.shape[0],) + out_shape
                    )
                    placeholder_idx = tuple(range(x.shape[-1]))
                    input_to_gate_map.update({i: (i,) for i in placeholder_idx})
                # extract the rows corresponding to encoding gates
                # thus those element to be combined with the jacobian
                # wrt the inputs
                d_encoding_angles = keras.ops.vstack(
                    [
                        d_angles[list(indices)]
                        for indices in zip(*input_to_gate_map.values())
                    ]
                )
                # discard the elements corresponding to encoding gates
                # to obtain only the part wrt the model's parameters
                indices_to_discard = reduce(tuple.__add__, input_to_gate_map.values())
                if tf.is_symbolic_tensor(d_angles):
                    # just some placeholder rows to continue symbolic computation
                    rows = [
                        d_angles[i]
                        for i in range(d_angles.shape[0] - len(indices_to_discard))
                    ]
                else:
                    rows = [
                        row
                        for i, row in enumerate(d_angles)
                        if i not in indices_to_discard
                    ]
                d_angles = keras.ops.reshape(
                    keras.ops.vstack(rows),
                    (-1, *out_shape),
                )
                # combine the jacobians wrt inputs with those
                # wrt the circuit angles
                d_x = keras.ops.einsum(
                    contraction,
                    jacobian_wrt_inputs,
                    d_encoding_angles,
                )
                d_x = keras.ops.einsum(f"{lhs},{rhs}", d_x, dy)
                d_x = keras.ops.reshape(d_x, x.shape)
            else:
                if tf.is_symbolic_tensor(d_angles):
                    d_angles = keras.ops.reshape(
                        d_angles, (jacobian.shape[0],) + out_shape
                    )
                d_x = None

            d_params = keras.ops.einsum(contraction, jacobian, d_angles)
            d_params = keras.ops.einsum(f"{lhs},{rhs}", d_params, dy)
            return d_x, d_params

        return y, grad
