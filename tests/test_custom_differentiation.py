import numpy as np
import pytest
import qibo
from qibo import Circuit, gates, hamiltonians
from qibo.backends import construct_backend

from qiboml.operations import expectation
from qiboml.operations.differentiation import parameter_shift, symbolical

BACKENDS = ["numpy", "tensorflow", "qibojit"]
DIFF_RULES = [symbolical, parameter_shift]


def build_circuit(angle):
    """Build parametric circuit and set rotation angle."""
    c = Circuit(1)
    c.add(gates.RX(0, 0.0))
    c.add(gates.M(0))
    c.set_parameters(angle)
    return c


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("differentiation_rule", DIFF_RULES)
def test_tf_chain_rule(backend, differentiation_rule, parameter):
    """..."""
    import tensorflow as tf  # pylint: disable=import-error

    qibo.set_backend("tensorflow")
    exec_backend = construct_backend(backend)

    # the RX angle is a function of the parameter
    angle = 2.0 * tf.math.log(tf.Variable(1.5))

    with tf.GradientTape() as tape:
        c = build_circuit(angle)
        h = hamiltonians.Z(1)
        expval = expectation.expectation(
            observable=h,
            circuit=c,
            differentiation_rule=differentiation_rule,
            exec_backend=exec_backend,
        )

    dexp_tf = tape.gradient(expval, angle)

    # analytical derivative
    dang_an = 2.0 / parameter
    dexp_an = np.sin(2.0 * np.log(parameter)) * dang_an

    np.testing.assert_allclose(dexp_tf, dexp_an, atol=1e-10)
