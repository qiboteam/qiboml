Switching the Autodifferentiation engine
----------------------------------------

The autodifferentiation engine is the object responsible for the calculation of the gradients for your model. It is closely co-ordinated with the backend and the interface to bridge the execution and training of the model seamlessly.

In addition to the native differentiation engines that come with ``torch`` and ``tensorflow``, in ``qiboml`` we provide two additional custom differentiation methods. A ``jax`` based one, :py:class:`qiboml.operations.differentiation.Jax`, particularly useful for enabling gradients calculation for simulation backends that do not support it, and a hardware compatible engine based on the `Parameter Shift Rule` (:py:class:`qiboml.operations.differentiation.PSR`), which is useful in general for simulation purposes as well, but more specifically to enable gradient calculation with hardware backends such as ``qibolab``. Keep in mind, however, that the ``PSR`` differentiation only supports one dimensional outputs, such as expectation values.

To set the differentiation engine for your own model you just have to set its ``differentiation`` attribute:

.. code::

   from qiboml.operations.differentiation import Jax, PSR

   # to use jax
   quantum_model = QuantumModel(encoding, circuit, decoding, differentiation=Jax)
   # to use PSR
   quantum_model = QuantumModel(encoding, circuit, decoding, differentiation=PSR)

If you don't specify any ``differentiation`` engine a default one is going to be used, depending on the in interface and backend in use. The table below summarize the default choices:

.. list-table::
   :widths: 25 25 50
   :header-rows: 1

   * - Interface
     - Backend
     - Differentiation
   * - torch
     - torch
     - torch
   * - torch
     - tensorflow / jax
     - Jax
   * - torch
     - other
     - PSR
   * - keras
     - tensorflow
     - tensorflow
   * - keras
     - torch / jax
     - Jax
   * - keras
     - other
     - PSR
