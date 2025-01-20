Defining a custom Encoder
-------------------------

An `Encoder` is a layer that takes care of embedding some classical data in a quantum circuit. Therefore, it represents the entry step necessary to enable data processing in a quantum fashion.

For instance, basic examples of this can be found in the :class:`qiboml.models.encoding.BinaryEncoding`, which takes a bitstring as input and simply encodes it in a set of ``X`` gates in the quantum circuit, or the :class:`qiboml.models.encoding.PhaseEncoding`, that maps an array of real data in rotation angles of some ``RX``, ``RY`` or ``RZ`` gates of the circuit.

Hence, broadly speaking, the quantum `Encoder` is a function :math:`f_e: \mathbf{x}\in\mathbb{R} \rightarrow C` that maps an input array of floats :math:`\mathbf{x}`, be it a ``torch.Tensor`` for the ``qiboml.interfaces.pytorch.QuantumModel`` or a ``tensorflow.Tensor`` for the ``qiboml.interfaces.keras.QuantumModel``, to an instance of a ``qibo.Circuit`` :math:`C`.

To define a custom encoder, then, one has only to write a custom function like that. For example, (provide code example) ...
