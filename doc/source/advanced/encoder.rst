Defining a custom Encoder
-------------------------

An `Encoder` is a layer that takes care of embedding some classical data into a quantum circuit. Therefore, it represents the entry step necessary to enable data processing in a quantum fashion.

For instance, basic examples of this can be found in the :py:class:`qiboml.models.encoding.BinaryEncoding`, which takes a bitstring as input and simply encodes it in a set of ``X`` gates in the quantum circuit, or the :py:class:`qiboml.models.encoding.PhaseEncoding`, that maps an array of real data in rotation angles of some ``RY`` gates of the circuit.

Hence, broadly speaking, the quantum `Encoder` is a function :math:`f_e: \mathbf{x}\in\mathbb{R}^n \rightarrow C` that maps an input array of :math:`n` floats :math:`\mathbf{x}`, be it a ``torch.Tensor`` for the :py:class:`qiboml.interfaces.pytorch.QuantumModel` or a ``tensorflow.Tensor`` for the :py:class:`qiboml.interfaces.keras.QuantumModel`, to an instance of a ``qibo.Circuit`` :math:`C`.

To define a custom encoder, ``qiboml`` handily provides an abstract :py:class:`qiboml.models.encoding.QuantumEncoding` class to inherit from. Let's say for example, that we had some heterogeneous data consisting of both real :math:`x_{\rm real}` and binary :math:`x_{\rm bin}` data stacked on top of each other in a single array

.. math::

   \mathbf{x} = x_{\rm real} \lvert x_{\rm bin}\;.

To encode at the same time these two type of data we could define a mixture of the :py:class:`qiboml.models.encoding.BinaryEncoding` and  :py:class:`qiboml.models.encoding.PhaseEncoding` encoders:

.. testcode::

   import numpy as np
   from qibo import gates
   from qiboml.models.encoding import QuantumEncoding

   class HeterogeneousEncoder(QuantumEncoding):

       def __init__(self, nqubits: int, real_part_len: int, bin_part_len: int):
           if real_part_len + bin_part_len != nqubits:
	       raise RuntimeError(
	       "``real_part_len`` and ``bin_part_len`` don't sum to ``nqubits``."
	       )

	   # use the general setup for a QuantumEncoding layer
	   # which mainly initialize an empty n-qubits circuit (self.circuit)
	   # and the set of qubits it insists on (by default self.qubits=range(nqubits))
	   super().__init__(nqubits)

	   self.real_qubits = self.qubits[:real_part_len]
	   self.bin_qubits = self.qubits[real_part_len:]

       def __call__(self, x: "ndarray") -> "Circuit":
           # check that the data is binary
           if not all((x[1] == 0) | (x[1] == 1)):
	       raise RuntimeError("Received non binary data")

	   # copy the internal circuit as we don't want to modify that
	   # every time a new input is processed
           circuit = self.circuit.copy()

	   # encode the real data
	   # the first row of x contains the real data
           for qubit, value in zip(self.real_qubits, x[0]):
               circuit.add(gates.RY(qubit, theta=value, trainable=False))

	   # encode the binary data
	   # the second row contains the binary data
	   for qubit, bit in zip(self.bin_qubits, x[1]):
               circuit.add(gates.RX(qubit, theta=bit * np.pi, trainable=False))

           return circuit

This way we have defined our custom layer that encodes the real data in the first ``real_part_len`` qubits of a quantum circuit, and the binary data in the second ``bin_part_len`` qubits.

In addition to this, we can optionally specify whether our custom encoder is differentiable or not. Namely, whether to calculate the derivatives with respect to its inputs upon differentiation. This is useful mostly for the sake of backpropagating the gradients to other layers that are found before the ``QuantumModel``, if any, which thus will compose their own gradients with the one coming from the ``QuantumModel``. As it will be discussed in more detail in the next section, this is crucial, for instance, to build trainable encoding layers.

The abstract :py:meth:`qiboml.models.encoding.QuantumEncoding` provides a property :py:meth:`qiboml.models.encoding.QuantumEncoding.differentiable` to set the differentiability of an encoder. It is set to ``True`` by default, but can be easily overridden by redifining it:

.. code::

   @property
   def differentiable(self) -> bool:
       if is_my_encoder_differentiable:
           return True
       return False

Keep in mind that, when ``differentiable`` is set to ``False``, all the gradients of the ``QuantumModel`` with respect to the inputs :math:`x` are going to automatically set to zero in the differentiation step.

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
