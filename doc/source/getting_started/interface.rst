Switching ML interface
----------------------

In addition to the ``torch`` interface (:class:`qibmol.interfaces.pytorch.QuantumModel`) showcased in the previous section, a ``keras`` based interface is available as well (:class:`qibmol.interfaces.keras.QuantumModel`). Similarly to ``torch``, this exposes a ``keras.Model`` object that can be easily integrated in ``keras`` pipelines.

In order to switch between the two interfaces, one can just make use of the corresponding ``QuantumModel`` object:

.. testcode::

   import qiboml.interfaces.pytorch as pt
   import qiboml.interfaces.keras as ks

   # the torch interface
   pt.QuantumModel

   # the keras interface
   ks.QuantumModel

Note that, since the pre-defined encoding, decoding and ansatz layers provided in ``qiboml`` are meant to be interface agnostic, they can be used indiscriminately with either one of the two interfaces:

.. testcode::

   from qiboml.models.encoding import BinaryEncoding
   from qiboml.models.decoding import Probabilities
   from qiboml.models.ansatze import ReuploadingCircuit

   # these are interface agnostic
   encoding = BinaryEncoding(2)
   decoding = Probabilties(2)
   circuit = ReuploadingCircuit(2)

   # build the torch model
   torch_model = pt.QuantumModel(encoding, circuit, decoding)
   # build the keras model
   keras_model = ks.QuantumModel(encoding, circuit, decoding)
