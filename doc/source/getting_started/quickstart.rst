Quick start
-----------

In order to build a Quantum Model you need to define three fundamental ingredients:

* An ``Encoder`` which takes care of embedding classical data inside of a Quantum Circuit.
* (`Optionally`) A parametrized quantum ``Circuit`` defining the actual computation the model will perform.
* A ``Decoder`` in charge of decoding the quantum information contained in the final state.

Such that a single evaluation of the model, divided in the three steps `Encoding` -> `Computation` -> `Decoding`, takes as input classical data and outputs classical data once again.

In ``qiboml`` we provide some standard pre-defined encoding and decoding layers, whereas the `Computation` part can be delegated to any ``qibo`` circuit (some standard quantum circuit ansatze are available as well). Therefore, building a ``qiboml`` model is rather immediate.

For instance using the ``torch`` interface:

.. testcode::

   import torch
   from qibo import Circuit, gates
   from qiboml.models.encodings import PhaseEncoding
   from qiboml.models.decodings import Probabilities
   from qiboml.interfaces.pytorch import QuantumModel

   encoding = PhaseEncoding(nqubits=3)
   decoding = Probabilities(nqubits=3)
   circuit = Circuit(3)
   circuit.add((gates.RZ(i, theta=torch.pi * torch.randn(1)) for i in range(3)))

   quantum_model = QuantumModel(encoding, circuit, decoding)
