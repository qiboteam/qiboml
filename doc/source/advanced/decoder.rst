Defining a custom Decoder
-------------------------

The `Decoder` is the part of the model in charge of transforming back the quantum information contained in a quantum state, to some classical representation consumable by classical calculators. This, in practice, translates to: obtaining the final quantum state, first, and, second, performing any suitable postprocessing onto it.

A very simple decoder, for instance, is the :py:class:`qiboml.models.decoding.Probabilities`, which extracts the probabilities from the final state. Similarly, :py:class:`qiboml.models.decoding.Samples` and :py:class:`qiboml.models.decoding.Expectation` respectively reconstruct the measured samples and calculate the expectation value of an observable on the final state.

Hence, the decoder is a function :math:`f_d: C \rightarrow \mathbf{y}\in\mathbb{R}^n`, that expects as input a ``qibo.Circuit``, executes it and finally perform some operation on the obtained final state to recover some classical output data :math:`\mathbb{y}` in the form of a float array.

``qiboml`` provides an abstract :py:class:`qiboml.models.decoding.QuantumDecoding` object which can be subclassed to define custom decoing layers. Let's say, for instance, that we would like to calculate the expectation value of a Transverse field Ising model (TFIM):

.. math::
        H = - \sum _{k=0}^{N} \, \left(Z_{k} \, Z_{k + 1} + h \, X_{k}\right) \, ,

on the final state of our system.
Handily ``qibo`` provides the implementation for the hamiltonian already, thus we only need to create a decoding layer that constructs the hamiltonian, executes the circuit and calculates the expecation value:

.. testcode::

   from qibo import Circuit
   from qiboml.models.decoding import QuantumDecoding
   from qibo.hamiltonian.models import TFIM

   class ExpectationTFIM(QuantumDecoding):

       def __init__(self, nqubits: int):
           super().__init__(nqubits)

	   # build the hamiltionan
	   self.observable = TFIM(nqubits)

       def __call__(self, x: Circuit):
           result = super().__call__(x)
	   return self.observable.expectation(result.state())

       @property
       def output_shape(self) -> tuple(int):
           (1, 1)

The ``super().__init__`` and ``super().__call__`` calls here are useful to simplify the implementation of the custom decoder. The ``super().__init__`` sets up the initial features needed, i.e. mainly an empty ``nqubits`` ``qibo.Circuit`` with a measurement appended on each qubit. Whereas, the ``super().__call__`` takes care of executing the ``qibo.Circuit`` passed as input ``x`` and returns a ``qibo.result`` object, hence one in ``(QuantumState, MeasurementOutcomes, CircuitResult)``.

In case you needed an even more fine-grained customization, you could always get rid of them and fully customize the initialization and call of the decoder. However, keep in mind that in order for a decoder to correctly work inside a ``qiboml`` pipeline, several components should be defined:

* A ``qibo`` compatible ``Backend``:

  if not manually specified, the :py:meth:`qiboml.models.decoding.QuantumDecoding.__init__` prepares the globally set backend and assigns it to its attribute :py:attr:`qiboml.models.decoding.QuantumDecoding.backend`, which is then used to execute the circuit inside of :py:meth:`qiboml.models.decoding.QuantumDecoding.__call__`. Therefore, make sure that at all times your custom decoder is provided with a valid ``Backend`` through its ``.backend`` attribute, and, moreover, that this backend choice is consistent in all the elements that care about it. For instance, in this example, even the ``TFIM`` object allows for backend specification and a mismatch between the decoder's and hamiltonian's backend may result in several problems.

.. code::

   class ExpectationTFIMCustomBackend(QuantumDecoding):

       # always use my custom backend for execution and
       # expectation value calculation
       def __init__(self, nqubits: int):
           self.backend = MyCustomBackend()

	   # build the hamiltionan, the backends should match!
	   self.observable = TFIM(nqubits, backend=self.backend)

       def __call__(self, x: Circuit):
           result = self.backend.execute_circuit(x)
	   return self.observable.expectation(result.state())

* Is it `analytically` differentiable?
