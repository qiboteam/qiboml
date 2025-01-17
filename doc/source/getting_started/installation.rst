Installation instructions
=========================

Installing with pip
"""""""""""""""""""

The installation using ``pip`` is the recommended approach to install Qiboml:

.. code-block:: bash

   pip install qiboml

Note that the base installation does not come with either ``torch`` or ``tensorflow``. You have to install those separately by yourself, unless you decide to install from source.

Installing from source
""""""""""""""""""""""

For development purposes or for having access to the most recent version of ``qiboml``, you can directly install from source:

.. code-block:: bash

   git clone https://github.com/qiboteam/qiboml.git
   cd qiboml
   pip install -e .

In this case you can also specify the extra dependencies as:

.. code-block:: bash

   pip install -e .[torch,tensorflow]
