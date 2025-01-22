Defining a custom Encoder
-------------------------

An `Encoder` is a layer that takes care of embedding some classical data in a quantum circuit. Therefore, it represents the entry step necessary to enable data processing in a quantum fashion.

For instance, basic examples of this can be found in the :class:`qiboml.models.encoding.BinaryEncoding`, which takes a bitstring as input and simply encodes it in a set of ``X`` gates in the quantum circuit, or the :class:`qiboml.models.encoding.PhaseEncoding`, that maps an array of real data in rotation angles of some ``RX``, ``RY`` or ``RZ`` gates of the circuit.

Hence, broadly speaking, the quantum `Encoder` is a function :math:`f_e: \mathbf{x}\in\mathbb{R} \rightarrow C` that maps an input array of floats :math:`\mathbf{x}`, be it a ``torch.Tensor`` for the ``qiboml.interfaces.pytorch.QuantumModel`` or a ``tensorflow.Tensor`` for the ``qiboml.interfaces.keras.QuantumModel``, to an instance of a ``qibo.Circuit`` :math:`C`.

To define a custom encoder, then, one only has to write a custom function like that. Let's take as an example the amplitude encoding, whch aims at embedding the input data in the amplitudes of our quantum state.

In particular, a practical realization of this can be achieved, for instance, through the Mottonen state preparation (`Mottonen et al. (2004) <https://arxiv.org/abs/quant-ph/0407010>`_). The idea is to prepare a set of controlled rotations which, under a suitable choice of angles, reproduce the desired amplitudes. In detail, the angles needed are given by the following expression:

.. math::

   \alpha_j^s = 2 \arcsin \frac{ \sqrt{\sum_{l=1}^{2^{s-1}} \mid a_{(2j-1)2^{s-1} + l}  \mid^2 } }{ \sqrt{\sum_{l=1}^{2^s} \mid a_{(j-1)2^s + l}  \mid^2 } }

TODO...
