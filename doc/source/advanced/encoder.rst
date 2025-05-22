Defining a custom Encoder
-------------------------

An `Encoder` is a layer that takes care of embedding some classical data into a quantum circuit. Therefore, it represents the entry step necessary to enable data processing in a quantum fashion.

For instance, basic examples of this can be found in the :py:class:`qiboml.models.encoding.BinaryEncoding`, which takes a bitstring as input and simply encodes it in a set of ``X`` gates in the quantum circuit, or the :py:class:`qiboml.models.encoding.PhaseEncoding`, that maps an array of real data in rotation angles of some ``RY`` gates of the circuit.

Hence, broadly speaking, the quantum `Encoder` is a function :math:`f_e: \mathbf{x}\in\mathbb{R}^n \rightarrow C` that maps an input array of :math:`n` floats :math:`\mathbf{x}`, be it a ``torch.Tensor`` for the :py:class:`qiboml.interfaces.pytorch.QuantumModel` or a ``tensorflow.Tensor`` for the :py:class:`qiboml.interfaces.keras.QuantumModel`, to an instance of a ``qibo.Circuit`` :math:`C`.

To define a custom encoder, ``qiboml`` handily provides an abstract :py:class:`qiboml.models.encoding.QuantumEncoding` class to inherit from. Let's say, for example, that we want to encode two-dimensional data ``x`` in our model and, for some reasons, we want our model to be more sensitive to the first variable (``x[0]``). To do so, we may be interested in constructing an encoder where ``x[0]`` is uploaded $n_1$ times, while ``x[1]`` is uploaded $n_2<n_1$ times.

.. testcode::

    from functools import cached_property
    import numpy as np
    from qibo import gates
    from qiboml.models.encoding import QuantumEncoding

    class CustomEncoder(QuantumEncoding):

    def __init__(self, nqubits: int, n1: int, n2: int):
        """
        Custom encoder where the first component of a two-dimensional data
        is uploaded ``n1`` times and the second component is uploaded ``n2`` times.
        """
        if n1 + n2 != nqubits:
            raise RuntimeError(
            "``n1`` and ``n2`` don't sum to ``nqubits``."
            )

        # use the general setup for a QuantumEncoding layer
        # which mainly initialize an empty n-qubits circuit (self.circuit)
        # and the set of qubits it insists on (by default self.qubits=range(nqubits))
        super().__init__(nqubits)

        self.x1_qubits = self.qubits[:n1]
        self.x2_qubits = self.qubits[(nqubits - n2):]

    def __call__(self, x: "ndarray") -> "Circuit":

        if len(x) != 2:
            raise ValueError(
            f"Data dimension is expected to be 2 in this custom encoder, while the received one is {len(x)}."
            )

        # copy the internal circuit as we don't want to modify that
        # every time a new input is processed
        circuit = self.circuit.copy()

        # encode the first component n1 times
        for qubit in self.x1_qubits:
            circuit.add(gates.RY(qubit, theta=x[0], trainable=False))

        # encode the second component n2 times
        for qubit in self.x2_qubits:
            circuit.add(gates.RY(qubit, theta=x[1], trainable=False))

        return circuit

This way we have defined our custom layer that encodes the first and the second component of a given two-dimensional data into respectively $n_1$ and $n_2$ gates.

In addition to this, we can optionally specify whether our custom encoder is differentiable or not. Namely, whether to calculate the derivatives with respect to its inputs upon differentiation. This is useful mostly for the sake of backpropagating the gradients to other layers that are found before the ``QuantumModel``, if any, which thus will compose their own gradients with the one coming from the ``QuantumModel``. As it will be discussed in more detail in the next section, this is crucial, for instance, to build trainable encoding layers.

The abstract :py:meth:`qiboml.models.encoding.QuantumEncoding` provides a property :py:meth:`qiboml.models.encoding.QuantumEncoding.differentiable` to set the differentiability of an encoder. It is set to ``True`` by default, but can be easily overridden by redifining it:

.. code::

   @property
   def differentiable(self) -> bool:
       if is_my_encoder_differentiable:
           return True
       return False

Keep in mind that, when ``differentiable`` is set to ``False``, all the gradients of the ``QuantumModel`` with respect to the inputs :math:`x` are going to automatically set to zero in the differentiation step.

Finally, there is an important property we have to set to allow the computation of gradients with respect to inputs in case hardware-compatible methods are chosen (parameter shift rule). We refer to the ``._data_to_gate`` method, where an ``encoding_map`` has to be defined in the form of a dictionary, where the keys are the indices corresponding to the components of the data and the value for each key is a list of integer numbers, corresponding to the list of gates where the target component of the data is encoded.

.. code::

    @cached_property
    def _data_to_gate(self):
        """
        Associate each data component with its index in the gates queue.
        In this case, we will follow the presented strategy, namely encoding ``x[0]``
        into the first ``n1`` qubits and ``x[1]`` in the second ``n2`` qubits.
        """
        return {
            "0": self.x1_qubits,
            "1": self.x2_qubits,
        }

On this way, we allow the custom differentiation rules to reconstruct the derivatives of any expectation value w.r.t. input data.

Trainable encoding layers
=========================

One thing that you probably noticed in the previous example, is that all the rotation gates we created in the circuit are set as ``trainable=False``. This is not a mistake but rather a precise design choice: all the eventual tuning of an encoding layer is delegated to external layers, i.e. the interface in practice, and not to the ``QuantumModel`` itself.

In other words, say that you wished to encode some data through a rotation gate as for the :py:class:`qiboml.models.encoding.PhaseEncoding`, but conditioned on some trainable parametrized function :math:`g`:

.. math::

   f_e = \rm{Encoding}_{g,\theta}(x)

one choice could be to make the function :math:`g` and the parameters :math:`\theta` part of the actual encoder thus something like:

.. code::

   def __init__(...):
       ...
       self.g = g
       self.theta = theta

   def __call__(x):
       x = self.g(x, self.theta)
       ...

however, this means that the burden of the gradients calculation

.. math::

   \frac{\partial \rm{Encoding}}{\partial x} = \frac{\partial \rm{Encoding}}{\partial g} \cdot \frac{\partial g}{\partial \theta} \cdot \frac{\partial \theta}{\partial x}

belongs to the ``QuantumModel``, which is problematic when, for instance, you use expensive hardware-compatible differentiation methods such as :py:class:`qiboml.operations.differentiation.PSR`. It is far easier and completely equivalent, instead, to move the parametrization of the encoding outside of the ``QuantumModel``, thus making the encoding a fixed transformation:

.. math::

   f_e = \rm{Encoding}(\;g(x,\theta)\;)\;.

In practice this means that any time you wish to parametrize the encoding step in any way, you should append to your model a layer that takes care of that just before the ``QuantumModel``, for instance:

.. code::

   # build your trainable transformation
   g = MyParametrizedTransformation(theta)
   # and stack it to the actual quantum model
   encoding_tunable_model = Sequential(
       g,
       quantum_model
   )
