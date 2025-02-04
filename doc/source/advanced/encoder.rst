Defining a custom Encoder
-------------------------

An `Encoder` is a layer that takes care of embedding some classical data in a quantum circuit. Therefore, it represents the entry step necessary to enable data processing in a quantum fashion.

For instance, basic examples of this can be found in the :py:class:`qiboml.models.encoding.BinaryEncoding`, which takes a bitstring as input and simply encodes it in a set of ``X`` gates in the quantum circuit, or the :py:class:`qiboml.models.encoding.PhaseEncoding`, that maps an array of real data in rotation angles of some ``RX``, ``RY`` or ``RZ`` gates of the circuit.

Hence, broadly speaking, the quantum `Encoder` is a function :math:`f_e: \mathbf{x}\in\mathbb{R}^n \rightarrow C` that maps an input array of floats :math:`\mathbf{x}`, be it a ``torch.Tensor`` for the :py:class:`qiboml.interfaces.pytorch.QuantumModel` or a ``tensorflow.Tensor`` for the :py:class:`qiboml.interfaces.keras.QuantumModel`, to an instance of a ``qibo.Circuit`` :math:`C`.

To define a custom encoder, ``qiboml`` handily provides an abstract :py:class:`qiboml.models.encoding.QuantumEncoding` class to inherit from. Let's say for example, that we had some heterogeneous data consisting of both real :math:`x_{real}` and binary :math:`x_{bin}` data stacked on top of each other in a single array

.. math::

   \mathbf{x} = x_{real} \lvert x_{bin}\;.

To encode at the same time these two type of data we could define a mixture of the :py:class:`qiboml.models.encoding.BinaryEncoding` and  :py:class:`qiboml.models.encoding.PhaseEncoding` encoders

.. testcode::

   import numpy as np
   from qiboml.models.encoding import QuantumEncoding

   class HeterogeneousEncoder(QuantumEncoding):

       def __init__(self, nqubits, real_part_len, bin_part_len):
           if real_part_len + bin_part_len != nqubits:
	       raise RuntimeError("``real_part_len`` and ``bin_part_len`` don't sum to ``nqubits``.")

	   super.__init__(nqubits)

	   self.real_qubits = self.qubits[:real_part_len]
	   self.bin_qubits = self.qubits[real_part_len:]

       def __call__(self, x) -> Circuit:
           # check that the data is binary
           if any(x[1] != 1 or x[1] != 0):
	       raise RuntimeError("Received non binary data")

           circuit = self.circuit.copy()

	   # the first row of x contains the real data
           for qubit, value in zip(self.real_qubits, x[0]):
               circuit.add(gates.RY(qubit, theta=value, trainable=False))

	   # the second row contains the binary data
	   for qubit, bit in zip(self.real_qubits, x[1]):
               circuit.add(gates.RX(qubit, theta=bit * np.pi, trainable=False))

        return circuit
