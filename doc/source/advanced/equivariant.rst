Building Equivariant Models
---------------------------

By default, when a trainable ``Circuit`` is passed to a ``QuantumModel`` through the ``circuit_structure``, all its parameters are considered independent. In some cases, however, one might want to just have a handful of independent parameters that are recombined in several different location inside the gates of your trainable quantum circuit.

For this reason ``Qiboml`` allows for the definition of custom circuits by means of user-defined functions, that can be added together with encoders and plain circuits to the ``circuit_structure``. For example, one can define a circuit with only three independent parameters that are recombined in several gates:

.. testcode::

   import torch

   # this defines 3 independent parameters
   def my_equivariant_circuit(theta, phi, lam):
      delta = 2 * torch.cos(phi) + lam**2
      gamma = lam * torch.exp(theta / 2)

      circuit = Circuit(2)
      circuit.add([gates.RZ(k, theta) for k in range(2)])
      circuit.add([gates.RX(k, lam) for k in range(2)])
      circuit.add([gates.RY(k, phi) for k in range(2)])
      circuit.add(gates.RZ(0, delta))
      circuit.add(gates.RX(1, gamma))

      return circuit

and then pass it to the ``QuantumModel`` through its ``circuit_structure`` argument:

.. code::

   from qiboml.interfaces.pytorch import QuantumModel

   model = QuantumModel(
      circuit_structure = [my_encoder, my_equivariant_circuit, my_plain_circuit],
      decoding = my_decoding,
   )

Note that, first, inside of your function you should use only operations that are compatible with the interface you are planning to use. Therefore, ``torch`` operations with the ``torch`` interface and ``keras`` or ``tensorflow`` operations with the ``keras`` interface. Second, that the each input to your custom function has to be a scalar, thus one argument is expected, for each independent parameter you wish to declare.
