Setup a quantum machine learning backend
========================================

Among other features, Qiboml is a backend provider for Qibo. In particular, it can
be used to construct differentiable backends to execute quantum machine learning
algorithms. Currently, we support Tensorflow automatic differentiation, while
Pytorch and Jax backends will be implemented in the next future.

To run a quantum algorithm in Qibo activating the tracing of gradients it is important
to set one of the backends provided by Qiboml.

.. code-block::

    import qibo
    from qibo import Circuit, gates, hamiltonians

    # set the backend using qiboml as provider
    qibo.set_backend(backend="qiboml", platform="tensorflow")
    tf = qibo.get_backend().tf

    # define some variables
    thetas = tf.Variable([0.4, 0.7])

    # construct a quantum circuit with Qibo
    c = Circuit(1)
    c.add(gates.H(q=0))
    c.add(gates.RZ(q=0, theta=0.))
    c.add(gates.RY(q=0, theta=0.))

    # construct an hamiltonian with Qibo
    h = hamiltonians.Z(nqubits=1)

    # automatic differentiation with tensorflow
    with tf.GradientTape() as tape:
        # set the variables w.r.t. we want the gradient into the circuit
        c.set_parameters(thetas)
        # compute an expectation value
        expval =  h.expectation(c().state())

    # collect gradient
    grad = tape.gradient(expval, thetas)
