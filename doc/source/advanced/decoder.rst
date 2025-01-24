Defining a custom Decoder
-------------------------

The `Decoder` is the part of the model in charge of transforming back the quantum information contained in a quantum state, to some classical representation consumable by classical calculators. This, in practice, translates to: obtaining the final quantum state, first, and, second, performing any suitable postprocessing onto it.

A very simple decoder, for instance, is the :py:class:`qiboml.models.decoding.Probabilities`, which extracts the probabilities from the final state. Similarly, :py:class:`qiboml.models.decoding.Samples` and :py:class:`qiboml.models.decoding.Expectation` respectively reconstruct the measured samples and calculate the expectation value of an observable on the final state.

Hence, the decoder is a function :math:`f_d: C \rightarrow \mathbf{y}\in\mathbb{R}^n`, that expects as input a ``qibo.Circuit``, executes it and finally perform some operation on the obtained final state to recover some classical output data :math:`\mathbb{y}` in the form of a float array.

``qiboml`` provides an abstract :py:class:`qiboml.models.decoding.QuantumDecoding` object which can be subclassed to define custom decoing layers. Let's say, for instance, that we would like to calculate the expectation value of a Transverse field Ising model (TFIM):

.. math::
        H = - \sum _{k=0}^{N} \, \left(Z_{k} \, Z_{k + 1} + h \, X_{k}\right) \, ,

on the final state of our system.
Handily ``qibo`` provides the implementation for the hamiltonian already, thus we only need to create a decoding layer that constructs the hamiltonian, executes circuit and calculates the expecation value:

.. testcode::

   from qiboml.models.decoding import QuantumDecoding
   from qibo.hamiltonian.models import TFIM

   class ExpectationTFIM(QuantumDecoding):

       def __init__(self, nqubits):
           super().__init__(nqubits)

	   # build the hamiltionan
	   self.observable = TFIM(nqubits)

       def __call__(self, x):
           final_state = super().__call__(x)
	   return self.observable.expectation(final_state)


TODO: add more comments on the backend used and what the abstract class does...
