==========================================
 Switching the Autodifferentiation engine
==========================================

The autodifferentiation engine is the object responsible for the calculation of the gradients for your model. It is closely co-ordinated with the backend and the interface to bridge the execution and training of the model seamlessly.

In addition to the native differentiation engines that come with ``torch`` and ``tensorflow``, in ``qiboml`` we provide two additional custom differentiation methods. A ``jax`` based one, :py:class:`qiboml.operations.differentiation.Jax`, particularly useful for enabling gradients calculation for simulation backends that do not support it, and a hardware compatible engine based on the `Parameter Shift Rule` (:py:class:`qiboml.operations.differentiation.PSR`), which is useful in general for simulation purposes as well, but more specifically to enable gradient calculation with hardware backends such as ``qibolab``. Keep in mind, however, that the ``PSR`` differentiation only supports one dimensional outputs, such as expectation values (:py:class:`qiboml.models.decoding.Expectation`).

To set the differentiation engine for your own model you just have to set its ``differentiation`` attribute:

.. code::

   from qiboml.operations.differentiation import Jax, PSR

   # to use jax
   quantum_model = QuantumModel(encoding, circuit, decoding, differentiation=Jax)
   # to use PSR
   quantum_model = QuantumModel(encoding, circuit, decoding, differentiation=PSR)

If you don't specify any ``differentiation`` engine a default one is going to be used, depending on the in interface and backend in use. The table below summarizes the default choices:


.. raw:: html

    <style>
    table, th, td {
    border: 1px solid black;
    text-align: center;
    }
    </style>
    <table style="width:80%" align="center">
        <thead>
            <tr>
                <th>Interface</th>
                <th>Backend</th>
                <th>Differentiation</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td rowspan="3">torch</td>
                <td>torch</td>
                <td>torch</td>
            </tr>
            <tr>
                <td>tensorflow / jax / numpy</td>
                <td>Jax</td>
            </tr>
            <tr>
                <td>other</td>
                <td>PSR</td>
            </tr>
            <tr>
                <td rowspan="3">keras</td>
                <td>tensorflow</td>
                <td>tensorflow</td>
            </tr>
            <tr>
                <td>torch / jax / numpy</td>
                <td>Jax</td>
            </tr>
            <tr>
                <td>other</td>
                <td>PSR</td>
            </tr>
        </tbody>
    </table>
