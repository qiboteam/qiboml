Using the Exact Geodesic Transport with Conjugate Gradients(EGT-CG) Optimizer
-------------------------------------------------------------------------------

The Exact Geodesic Transport with Conjugate Gradients (EGT-CG) optimizer is a curvature-aware Riemannian optimizer designed specifically for variational circuits based on the Hamming-weight encoder (HWE) ansatze.

It updates parameters along exact geodesic paths on the hyperspherical manifold defined by the HWE, combining analytic metric computation, conjugate-gradient memory, and dynamic learning rates for fast, globally convergent optimization.


For more details, see:

A. J. Ferreira‑Martins et al., *Variational quantum algorithms with exact geodesic transport*, 

`arXiv:2506.17395 (2025) <https://arxiv.org/abs/2506.17395>`_.


The current configuration focuses on ground state estimation problems where given a variational state :math:`\ket{\psi(\boldsymbol{\theta})}` parameterized by hyperspherical angles :math:`\boldsymbol{\theta}`, 

the loss function minimized is the energy expectation value of the given hamiltonian:

.. math::
    \mathcal{L}(\boldsymbol{\theta}) = \bra{\psi(\boldsymbol{\theta})} \, H \, \ket{\psi(\boldsymbol{\theta})} \, ,

But it is important to note that this framework generalizes to arbitrary loss functions.


Defining and running the optimizer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
In `qiboml`, the EGT-CG optimizer is implemented in the class
:class:`qiboml.models.optimizers.ExactGeodesicTransportCG`.

.. autoclass:: qiboml.models.optimizers.ExactGeodesicTransportCG
   :members:
   :undoc-members:
   :show-inheritance:

It requires as inputs the number of qubits, the Hamming weight, a Hamiltonian, and your initial parameters.

**Example usage:**

.. code-block:: python

    import numpy as np
    from qibo import hamiltonians
    from scipy.special import comb
    from qiboml.models.optimizers import ExactGeodesicTransportCG

    nqubits = 4
    weight = 2
    dim = int(comb(nqubits, weight))
    np.random.seed(42)

    # Initial angles in hyperspherical coordinates
    theta_init = np.random.uniform(low=0, high=np.pi, size=dim - 1)

    # Define the Hamiltonian
    hamiltonian = hamiltonians.XXZ(nqubits=nqubits)

    # Instantiate the optimizer
    optimizer = ExactGeodesicTransportCG(
        nqubits=nqubits,
        weight=weight,
        hamiltonian=hamiltonian,
        angles=theta_init,
    )

    # Run optimization for 20 steps
    final_loss, losses, final_params = optimizer(steps=20)

Available Arguments
~~~~~~~~~~~~~~~~~~

When constructing an :class:`ExactGeodesicTransportCG`, you may specify:

- ``nqubits``: Number of qubits in the circuit.
- ``weight``: Hamming weight of the state.
- ``hamiltonian``: :class:`qibo.hamiltonians.Hamiltonian` instance.
- ``angles``: Initial hyperspherical angles (numpy array).
- ``backtrack_rate``: Backtracking factor for line search (default: ``0.5``).
- ``geometric_gradient``: Whether to use geometric gradient or finite difference (default: ``False``).
- ``multiplicative_factor``: Scaling of initial learning rate (default: ``1.0``).
- ``c1``, ``c2``: Wolfe line search constants (defaults: ``1e-3``, ``0.9``).
- ``backend``: Optional qibo backend (default: global backend)
  
Implementation Details 
~~~~~~~~~~~~~~~~~~~~~~~
The optimizer constructs the state at each iteration using the :func:`qiboml.models.encodings.hamming_weight_encoder` circuit, computes the loss as the Hamiltonian expectation value, and updates the angles by:

1. Computing the Riemannian (natural) gradient.
2. Performing a conjugate-gradient step along the geodesic direction, with the step size determined by a Wolfe condition line search

At the end of the run, it returns:

- ``final_loss``: Loss at optimal parameters.
- ``losses``: List of losses per iteration.
- ``final_params``: Optimal angles.