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

Note that these three are just the backends that ``qiboml`` provides out of the box, but actually any ``qibo``-compatible backend can be used. In particular, for istance, in case you wished to try out your model on a real quantum hardware chip, you could just set up `qibolab <https://qibo.science/qibolab/stable/>`_ for running and not worry about anything else.
