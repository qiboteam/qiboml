Switching the Execution Backend
-------------------------------

In ``qiboml`` the backend is the object in charge of, both, executing the quantum circuit in the first place and handling any side processing needed for training and evaluating your model (e.g. calculation of probabilities, sampling, etc...).

``qiboml`` provides itself three different backends for computation that are compliant with the ``qibo`` standards: :py:class:`qiboml.backends.JaxBackend`, :py:class:`qiboml.backends.PyTorchBackend` and :py:class:`qiboml.backends.TensorflowBackend`.

Any one of them can be set up by using the usual ``set_backend`` method from ``qibo``:

.. testcode::

   from qibo import set_backend

   # set jax
   set_backend("qiboml", platform="jax")
   # set torch
   set_backend("qiboml", platform="pytorch")
   # set tensorflow
   set_backend("qiboml", platform="tensorflow")

this sets the backend globally, which means it will be automatically used everywhere it is needed.

Alternatively, you can also directly construct an instance of the backend you wish to use, remember, however, to explicitely set it in all the pieces that require a backend specification (e.g. the decoding layer):

.. testcode::

   from qiboml.backends import PyTorchBackend, TensorflowBackend
   from qiboml.models.decoding import Probabilities

   # construct a local istance of the backend
   backend = PyTorchBackend()
   # assign it to the decoding layer upon initialization
   decoding = Probabilities(nqubits=3, backend=backend)
   # you can even switch it after initialization,
   # however, pay always attention to consistency!
   tf_backend = TensorflowBackend()
   decoding.set_backend(tf_backend)

Note that three backends mentioned above are just those that ``qiboml`` provides out of the box, but actually any ``qibo``-compatible backend can be used. In particular, for istance, in case you wished to try out your model on a real quantum hardware chip, you could just set up `qibolab <https://qibo.science/qibolab/stable/>`_ for running and not worry about anything else.
