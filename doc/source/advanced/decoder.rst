Defining a custom Decoder
-------------------------

The `Decoder` is the part of the model in charge of transforming back the quantum information contained in a quantum state, to some classical representation consumable by classical calculators. This, in practice, translates to: obtaining the final quantum state, first, and, second, performing any suitable postprocessing onto it.

A very simple decoder, for instance, is the :py:class:`qiboml.models.decoding.Probabilities`, which extracts the probabilities from the final state. Similarly, :py:class:`qiboml.models.decoding.Samples` and :py:class:`qiboml.models.decoding.Expectation` respectively reconstruct the measured samples and calculate the expectation value of an observable on the final state.

Hence, the decoder is a function :math:`f_d: C \rightarrow \mathbf{y}\in\mathbb{R}^n`, that expects as input a ``qibo.Circuit``, executes it and finally perform some operation on the obtained final state to recover some classical output data :math:`\mathbb{y}` in the form of a float array.

``qiboml`` provides an abstract :py:class:`qiboml.models.decoding.QuantumDecoding` object which can be subclassed to define custom decoding layers. Let's say, for instance, that we would like to calculate the expectation values of two different observables:

.. math::
   O_{even} = Z_0 \otimes Z_2 ,\\
   O_{odd} = Z_1 \otimes Z_3

on the final state of our system, and measure how close the expectation values are:

.. math::
   d = \lvert \langle O_{even} \rangle - \langle O_{odd} \rangle \rvert.

To do this we only need to create a decoding layer that constructs the two observables upon initialization and, when called, executes the circuit and calculates the distance :math:`d`:

.. testcode::

   import numpy as np

   from qiboml.models.decoding import QuantumDecoding
   from qibo import Circuit
   from qibo.symbols import Z
   from qibo.hamiltonian import SymbolicHamiltonian

   class MyCustomDecoder(QuantumDecoding):

       def __init__(self, nqubits: int):
           super().__init__(nqubits)
	   # build the observables
	   self.o_even = SymbolicHamiltonian(Z(0)*Z(2), nqubits=nqubits)
	   self.o_odd = SymbolicHamiltonian(Z(1)*Z(3), nqubits=nqubits)

       def __call__(self, x: Circuit):
           # execute the circuit and collect the final state
           final_state = super().__call__(x).state()
	   # calculate the expectation values
	   exp_even = self.o_even.expectation(final_state)
	   exp_odd = self.o_odd.expectation(final_state)
	   # use numpy to calculate the distance
	   return np.abs(exp_even - exp_odd)

       # specify the shape of the output
       @property
       def output_shape(self) -> tuple(int):
           (1, 1)

Note that it is important to also specify what is the expected output shape of the decoder, for example as in this case we are just dealing with expectation values and, thus, scalars, we are going to set it as :math:`(1,1)`.

The ``super().__init__`` and ``super().__call__`` calls here are useful to simplify the implementation of the custom decoder. The ``super().__init__`` sets up the initial features needed, i.e. mainly an empty ``nqubits`` ``qibo.Circuit`` with a measurement appended on each qubit. Whereas, the ``super().__call__`` takes care of executing the ``qibo.Circuit`` passed as input ``x`` and returns a ``qibo.result`` object, hence one in ``(QuantumState, MeasurementOutcomes, CircuitResult)``.

In case you needed an even more fine-grained customization, you could always get rid of them and fully customize the initialization and call of the decoder. However, keep in mind that in order for a decoder to correctly work inside a ``qiboml`` pipeline, several components should be defined:

* A ``qibo`` compatible ``Backend``:

  if not manually specified, the :py:meth:`qiboml.models.decoding.QuantumDecoding.__init__` prepares the globally-set backend and assigns it to its attribute :py:attr:`qiboml.models.decoding.QuantumDecoding.backend`, which is then used to execute the circuit inside of :py:meth:`qiboml.models.decoding.QuantumDecoding.__call__`. Therefore, make sure that at all times your custom decoder is provided with a valid ``Backend`` through its ``.backend`` attribute, and, moreover, that this backend choice is consistent in all the elements that care about it. For instance, in this example, even the observables allow for backend specification and a mismatch between the decoder's and observables backends may result in several problems.

.. code::

   class MyCustomDecoderWithCustomBackend(QuantumDecoding):

       # always use my custom backend for execution and
       # expectation value calculation
       def __init__(self, nqubits: int):
           self.backend = MyCustomBackend()
	   # the backends should match!
	   self.o_even = SymbolicHamiltonian(Z(0)*Z(2), nqubits=nqubits)
	   self.o_odd = SymbolicHamiltonian(Z(1)*Z(3), nqubits=nqubits)

       def __call__(self, x: Circuit):
           final_state = self.backend.execute_circuit(x).state()
	   exp_even = self.o_even.expectation(final_state)
	   exp_odd = self.o_odd.expectation(final_state)
	   return np.abs(exp_even - exp_odd)

* A boolean ``analytic`` property:

  for differentiation purposes, it is important to know whether the decoding step is `analytically` differentiable, i.e. if any sampling is involved in practice. If no sampling is involved, all the operations can be easily tracked and the gradients can be analitically calculated via standard differentiation methods (native ``pytorch`` or ``jax`` for example). Otherwise, we must recurr to different ways for obtaining the gradients, such as the :py:class:`qiboml.operations.differentiation.PSR`. For this purpose, each decoding object has a ``analytic`` property that is set to ``True`` by default:

.. code::

   class MyCustomDecoder(QuantumDecoding):

       @property
       def analytic(self,) --> bool:
           if some_condition:
               return True
	   return False
