"""Keras interface to qiboml layers"""

# pylint: disable=no-member
# pylint: disable=unexpected-keyword-arg

import string
from dataclasses import dataclass
from functools import reduce
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
        def jac_functional(x):
            with tf.GradientTape() as tape:
                tape.watch(x)
                y = f(x)
            return tape.jacobian(y, x)

        return jac_functional

    @staticmethod
    def jacfwd(f: Callable, argnums: Union[int, Tuple[int]]) -> Callable:
        # no available implementation of forward differentiation in tensorflow
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
        """
        for circ in self.circuit_structure:
            if isinstance(circ, QuantumEncoding):
                with tf.GradientTape() as tape:
                    tape.watch(x)
                    y = circ(x)
                    y = tf.stack([
                        p
                        for pars in y.get_parameters(
                                include_not_trainable=True
                        )
                        for p in pars
                    ])
                grad = tape.gradient(y, x)

                #grad = self.jacobian_functionals[id(circ)](x)
                if grad is not None:
                    return True
                break
        """
        # return hasattr(x, "op") and len(x.op.inputs) > 0
        return True


@dataclass(eq=False)
class QuantumModel(keras.Model):  # pylint: disable=no-member
    """
    The Keras interface to qiboml models.

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

    def __post_init__(self):
        super().__init__()

        if isinstance(self.circuit_structure, Circuit):
            self.circuit_structure = [self.circuit_structure]

        params = utils.get_params_from_circuit_structure(self.circuit_structure)
        params = keras.ops.cast(
            self.backend.to_numpy(params),
            # params,
            "float64",
        )  # pylint: disable=no-member
        self.circuit_parameters = self.add_weight(
            shape=params.shape, initializer="zeros", trainable=True, dtype="float64"
        )
        self.set_weights([params])

        if self.circuit_tracer is None:
            self.circuit_tracer = KerasCircuitTracer
        self.circuit_tracer = self.circuit_tracer(self.circuit_structure)

        if self.differentiation is None:
            self.differentiation = utils.get_default_differentiation(
                decoding=self.decoding,
                instructions=DEFAULT_DIFFERENTIATION,
            )
        """
        if self.differentiation is not None:
            self.custom_gradient = QuantumModelCustomGradient(
                self.circuit_structure,
                self.decoding,
                self.backend,
                self.differentiation,
            )
        """
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
        """
        x_is_not_None = x.shape[0] != 0

        # check whether we have to derive wrt inputs
        if x_is_not_None and tf.is_symbolic_tensor(x):

            self.wrt_inputs = (
                self.circuit_tracer.is_encoding_differentiable
                and self.circuit_tracer.requires_gradient(x)
            )
        """

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
        # check output shape of decoding layers, their returned tensor
        # shape should match the output_shape attribute and this should
        # not be necessary (only useful for symbolic execution)!!
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
            if tf.is_symbolic_tensor(d_angles):
                # breakpoint()
                d_angles = keras.ops.reshape(d_angles, (jacobian.shape[0],) + out_shape)
            # contraction to combine jacobians wrt inputs/parameters with those
            # wrt the circuit angles
            contraction = ((0, 1), (0,) + tuple(range(2, len(out_shape) + 2)))
            contraction = ",".join(
                "".join(string.ascii_letters[i] for i in indices)
                for indices in contraction
            )
            # contraction to combine with the gradients coming from outside
            """
            indices = tuple(range(len(dy.shape)))
            lhs = "".join(string.ascii_letters[i] for i in indices)
            rhs = string.ascii_letters[len(indices)] + lhs
            """
            rhs = "".join(
                string.ascii_letters[i] for i in tuple(range(1, len(dy.shape) + 1))
            )
            lhs = string.ascii_letters[0] + rhs

            if jacobian_wrt_inputs is not None:
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
                    # breakpoint()
                    # just some placeholder rows to continue symbolic computation
                    rows = [
                        d_angles[i]
                        for i in range(jacobian.shape[0] - len(indices_to_discard))
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
                # tmp = lhs + "".join(
                #    string.ascii_letters[i] for i in range(len(indices), len(d_x.shape))
                # )
                # d_x = keras.ops.einsum(f"{lhs},{tmp}", d_x, dy)
                d_x = keras.ops.einsum(f"{lhs},{rhs}", d_x, dy)
            else:
                d_x = None

            """
            if x_is_not_None:
                # double check this
                # the reshape here should be needed for symbolic execution only
                d_x = keras.ops.reshape(d_x, dy.shape + x.shape)
            """
            d_params = keras.ops.einsum(contraction, jacobian, d_angles)
            # d_params = keras.ops.reshape(d_params, tuple(params.shape) + dy.shape)
            d_params = keras.ops.einsum(f"{lhs},{rhs}", d_params, dy)
            return d_x, d_params

        return y, grad
