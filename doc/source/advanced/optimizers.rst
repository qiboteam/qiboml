Using the Exact Geodesic Transport with Conjugate Gradients(EGT-CG) Optimizer
-------------------------------------------------------------------------------

The Exact Geodesic Transport with Conjugate Gradients (EGT-CG) optimizer is a curvature-aware Riemannian optimizer designed specifically for variational circuits based on the Hamming-weight encoder (HWE) ansatz (see  Farias et al., *Quantum encoder for fixed-Hamming-weight subspaces*, `Phys. Rev. Applied 23, 044014 (2025) <https://doi.org/10.1103/PhysRevApplied.23.044014>).

It updates parameters along exact geodesic paths on the hyperspherical manifold defined by the HWE, combining analytic metric computation, conjugate-gradient memory, and dynamic learning rates for fast, globally convergent optimization.


For more details, see:

Ferreira-Martins et al., *Quantum optimization with exact geodesic transport*, `arXiv:2506.17395 (2025) <https://arxiv.org/abs/2506.17395>`_.

The optimizer works with an ansatz :math:`\ket{\psi(\boldsymbol{\theta})}` parameterized by hyperspherical angles :math:`\boldsymbol{\theta}`, and the implementation allows one to work with arbitrary loss functions. VQE is achieved by specifying the loss function :math:`\mathcal{L}(\boldsymbol{\theta}) = \bra{\psi(\boldsymbol{\theta})} \, H \, \ket{\psi(\boldsymbol{\theta})}` and passing the hamiltonian as one of its arguments, as can be seen in the example below. 


Defining and running the optimizer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
In `qiboml`, the EGT-CG optimizer is implemented in the class :class:`qiboml.models.optimizers.ExactGeodesicTransportCG`.

.. autoclass:: qiboml.models.optimizers.ExactGeodesicTransportCG
   :members:
   :undoc-members:
   :show-inheritance:

Example usage - VQE
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from qibo import hamiltonians, set_backend, get_backend
    from qiboml.models.optimizers import ExactGeodesicTransportCG

    set_backend("qiboml", platform="pytorch")
    backend = get_backend()

    nqubits = 4
    weight = nqubits // 2
    hamiltonian = hamiltonians.XXZ(nqubits=nqubits, delta=0.5, backend=backend).matrix
    loss_fn = "exp_val"

    def make_callback(print_every):
        def callback(
            iter_num,
            loss,
            **kwargs,
        ):
            if iter_num % print_every == 0:
                print(f"Iter {iter_num}: loss = {loss:.6f}")

        return callback

    optimizer = ExactGeodesicTransportCG(
        nqubits=nqubits,
        weight=weight,
        loss_fn=loss_fn,
        loss_kwargs={"hamiltonian": hamiltonian},
        initial_parameters=None,
        backtrack_rate=0.5,
        backtrack_multiplier=1.5,
        backtrack_min_lr=1e-6,
        c1=0.485,
        c2=0.999,
        callback=make_callback(print_every=1),
        seed=13,
        backend=backend,
    )
    final_loss, losses, final_params = optimizer(steps=20)



Available arguments
~~~~~~~~~~~~~~~~~~

When constructing an :class:`ExactGeodesicTransportCG`, you may specify:

- ``nqubits``: Number of qubits in the circuit.
- ``weight``: Hamming weight of the state/desired subspace.
- ``loss_fn``: Loss function to be optimized. It can be either a callable specifying the loss, or the string `"exp_val"`, which fixes the loss for the usual VQE. If it's a callable, make sure that the first two arguments are the circuit to be executed and the execution backend, respectively, as shown in the example below.
- ``loss_kwargs``: Dictionary where you can pass arguments of the loss as kwargs, apart from the two mandatory (circuit and backend). If you set the loss `"exp_val"`, you must pass the hamiltonian here as an item `{'hamiltonian': hamiltonian}`. It may be expressed either as a dense or sparse matrix.
- ``initial_parameters``: Initial hyperspherical angles (parameters of the circuit). If `None`, parameters are initialized from a Haar-random state (seed controls reproducibility).
- ``backtrack_rate``: Backtracking factor for conjugate learning rate backtrack search.
- ``backtrack_multiplier``: Scaling factor applied to the initial learning rate for the backtrack. Usually, it's greater than 1 to guarantee a wider search space.
- ``backtrack_min_lr``: Minimum learning rate to be tested in the backtrack.
- ``c1``, ``c2``: Wolfe line search constants.
- ``callback``: Callable for callback.
- ``seed``: Random seed.
- ``backend``: Optional qibo backend (default: global backend). If you set `"exp_val"` as the loss, you can use the `numpy` backend, which will be the fastest option as backpropagation will not be needed. If a callable is passed for a generic loss, you must use `qiboml` backend with the prefered platform (`"pytorch"`, `"tensorflow"` or `"jax"`)for backpropagation.

As an example, if you'd like to explicitly define the expectation value as the loss function, you could do as follows:

.. code-block:: python

    def loss_func_expval(circuit, backend, hamiltonian) -> float:
        psi = backend.execute_circuit(circuit).state()
        return backend.real(backend.conj(psi) @ hamiltonian @ psi)

    loss_fn = loss_func_expval
  
At the end of the run, the following objects are returned:

- ``final_loss``: Loss at final parameters.
- ``losses``: List of losses per epoch.
- ``final_params``: Final parameters.

Also, you can access the arttributes `n_calls_loss` and `n_calls_gradient`, which store respectively the number of times that the loss and the gradient were computed during the optimization.