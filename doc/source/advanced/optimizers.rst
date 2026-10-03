Using the Exact Geodesic Transport with Conjugate Gradients(EGT-CG) Optimizer
-------------------------------------------------------------------------------

The Exact Geodesic Transport with Conjugate Gradients (EGT-CG) optimizer is a curvature-aware
Riemannian optimizer designed specifically for variational circuits based on the Hamming-weight
encoder (HWE) ansatz (see  Farias et al., *Quantum encoder for fixed-Hamming-weight subspaces*,
`Phys. Rev. Applied 23, 044014 (2025) <https://doi.org/10.1103/PhysRevApplied.23.044014>`_).

It updates parameters along exact geodesic paths on the hyperspherical manifold defined by the HWE,
combining analytic metric computation, conjugate-gradient memory, and dynamic learning rates
for fast, globally convergent optimization.

For more details, see: Ferreira-Martins et al., *Quantum optimization with exact geodesic
transport*, `arXiv:2506.17395 (2025) <https://arxiv.org/abs/2506.17395>`_.

The optimizer works with an ansatz :math:`\ket{\psi(\boldsymbol{\theta})}` parameterized by
hyperspherical angles :math:`\boldsymbol{\theta}`, and the implementation allows one to work with
arbitrary loss functions. VQE is achieved by specifying the loss function
:math:`\mathcal{L}(\boldsymbol{\theta}) = \bra{\psi(\boldsymbol{\theta})} \, H \, \ket{\psi(\boldsymbol{\theta})}`
and passing the hamiltonian as one of its arguments, as can be seen in the example below.

In ``qiboml``, the EGT-CG optimizer is implemented in the class
:class:`qiboml.models.optimizers.ExactGeodesicTransportCG`.

Example usage - VQE for Heisenberg XXZ model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from qibo import get_backend, set_backend
    from qibo.hamiltonians import XXZ
    from qiboml.models.optimizers import ExactGeodesicTransportCG

    set_backend("qiboml", platform="pytorch")
    backend = get_backend()

    nqubits = 4
    weight = nqubits // 2
    hamiltonian = XXZ(nqubits=nqubits, delta=0.5, backend=backend).matrix
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

When constructing an :class:`qiboml.models.optimizers.ExactGeodesicTransportCG` object,
the arguments are:

- ``nqubits``: Number of qubits in the circuit.
- ``weight``: Hamming weight of the state/desired subspace.
- ``loss_fn``: Loss function to be optimized. It can be either a callable specifying the loss,
    or the string ``"exp_val"``, which fixes the loss for the usual VQE. If it's a callable,
    make sure that the first two arguments are the circuit to be executed and the execution backend,
    respectively, as shown in the example below.
- ``loss_kwargs``: Dictionary where one can pass arguments of the loss as kwargs, apart from the
    two mandatory (circuit and backend). If one sets the loss as ``"exp_val"``, hamiltonian must be
    passed here as an item ``{'hamiltonian': hamiltonian}``. It may be expressed either as a dense
    or sparse matrix.
- ``initial_parameters``: Initial hyperspherical angles (parameters of the circuit). If ``None``,
    parameters are initialized from a Haar-random state (seed controls reproducibility).
- ``backtrack_rate``: Backtracking factor for conjugate learning rate backtrack search.
- ``backtrack_multiplier``: Scaling factor applied to the initial learning rate for the backtrack.
    Usually, it's greater than 1 to guarantee a wider search space.
- ``backtrack_min_lr``: Minimum learning rate to be tested in the backtrack.
- ``c1``, ``c2``: Wolfe line search constants.
- ``callback``: Callable for callback.
- ``seed``: Random seed.
- ``backend``: Optional qibo backend (default: global backend). If one sets ``"exp_val"`` as the
    loss, the ``numpy`` backend (in particular, via the
    `Hamming-weight simulator <https://https://qibo.science/qibo/stable/api-reference/qibo.html#simulation-of-hamming-weight-preserving-circuits>`_)
    can be used, which will be the fastest option as backpropagation will not be needed.
    If a callable is passed for a generic loss, one must use the ``qiboml`` backend with the
    prefered platform (``"pytorch"``, ``"tensorflow"`` or ``"jax"``) for backpropagation.
    We still recommend the use of the Hamming-weight Backend for increased efficiency.

As an example, if one wishes to explicitly define the expectation value as the loss function,
one does as follows:

.. code-block:: python

    def loss_func_expval(circuit, backend, hamiltonian) -> float:
        psi = backend.execute_circuit(circuit).state()
        return backend.real(backend.conj(psi) @ hamiltonian @ psi)

    loss_fn = loss_func_expval

At the end of the run, the following objects are returned:

- ``final_loss``: Loss at final parameters.
- ``losses``: List of losses per epoch.
- ``final_params``: Final parameters.

Also, one can access the attributes ``n_calls_loss`` and ``n_calls_gradient``, which store
respectively the number of times that the loss and the gradient were computed during the
optimization.


Using the Quantum Natural Gradient (QNG) Optimizer
--------------------------------------------------

The Quantum Natural Gradient (QNG) optimizer is a gradient-based optimizer for parameterized
circuits that takes into account the geometry of the space of quantum states (see Stokes et al.,
*Quantum Natural Gradient*, `Quantum 4, 269 (2020) <https://doi.org/10.22331/q-2020-05-25-269>`_).
Instead of following the plain gradient :math:`\nabla \mathcal{L}` of the loss function
:math:`\mathcal{L}(\boldsymbol{\theta})`, where :math:`\boldsymbol{\theta}` are the circuit
parameters, it rescales it with the inverse of the Fubini-Study metric tensor :math:`g`,
which measures how much the state :math:`\ket{\psi(\boldsymbol{\theta})}` changes when each
parameter is varied:

.. math::

    \boldsymbol{\theta} \leftarrow \boldsymbol{\theta} - \eta \, (g + \lambda I)^{-1} \,
        \nabla \mathcal{L}(\boldsymbol{\theta}) \, ,

where :math:`\eta` is the learning rate, :math:`I` is the identity matrix, and :math:`\lambda`
is a small regularization strength that keeps the inversion stable when :math:`g` is singular.

In ``qiboml``, the QNG optimizer is implemented in the class
:class:`qiboml.models.optimizers.QuantumNaturalGradient`.
The loss is the expectation value of an observable, and it is defined by passing a
:class:`qiboml.models.decoding.Expectation` decoding, which also sets the backend.

Supported gates
~~~~~~~~~~~~~~~

Internally, the circuit is decomposed into fixed gates and single-parameter Pauli rotations,
whose angles are linear functions of the parameters of the original gates.
For instance, :math:`U_{3}(\theta, \phi, \lambda) = R_{Z}(\phi) R_{Y}(\theta) R_{Z}(\lambda)`,
where :math:`R_{Z}` and :math:`R_{Y}` are rotations around the :math:`Z` and :math:`Y` axes.
Derivatives are computed with respect to the angles of the rotations and mapped back to the
parameters of the original circuit with the chain rule, which also takes care of gates whose
decomposition contains several rotations that depend on the same parameter, such as
:class:`qibo.gates.RBS`. The circuit passed by the user is not modified, apart from its parameters
being updated during the optimization.

The supported trainable gates are:

- single-qubit gates: :class:`qibo.gates.RX`, :class:`qibo.gates.RY`, :class:`qibo.gates.RZ`,
    :class:`qibo.gates.U1`, :class:`qibo.gates.U2` and :class:`qibo.gates.U3`;
- multi-qubit gates: :class:`qibo.gates.RBS`, :class:`qibo.gates.GIVENS`, :class:`qibo.gates.RXX`,
    :class:`qibo.gates.RYY`, :class:`qibo.gates.RZZ`, :class:`qibo.gates.RZX`,
    :class:`qibo.gates.RXXYY`, :class:`qibo.gates.CRX`, :class:`qibo.gates.CRY`,
    and :class:`qibo.gates.CRZ`.

Other trainable gates, such as :class:`qibo.gates.fSim`, raise a ``NotImplementedError``.
The parameters are ordered gate by gate, as :math:`(\theta, \phi, \lambda)` for each
:class:`qibo.gates.U3` gate.

Choosing the differentiation engine
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The gradient of the loss is computed with one of the differentiation engines of ``qiboml``,
which is selected with the argument ``differentiation``:

- :class:`qiboml.differentiations.psr.PSR` (default): parameter-shift rule;
- :class:`qiboml.differentiations.adjoint.Adjoint`: adjoint differentiation;
- :class:`qiboml.differentiations.jax.Jax`: automatic differentiation with ``jax``.

By default, the metric tensor is computed exactly from the state vector of the circuit.
It needs the derivatives of the state vector. These are computed with the
:class:`qiboml.differentiations.jax.Jax` engine if it is the chosen one, while the other engines
only differentiate expectation values, and the derivatives of the state are obtained with the
parameter-shift rule applied to the state vector.

Sampled metric tensor
~~~~~~~~~~~~~~~~~~~~~

The metric tensor can also be estimated from measurements, by passing the number of shots as
``metric_nshots``. In this case, the entries of the metric tensor are obtained with the
parameter-shift rule applied to the overlap
:math:`F(\boldsymbol{\theta}, \boldsymbol{\theta}') = |\braket{\psi(\boldsymbol{\theta}) | \psi(\boldsymbol{\theta}')}|^{2}`
between the states prepared by the decomposed circuit at two sets of angles
:math:`\boldsymbol{\theta}` and :math:`\boldsymbol{\theta}'`, using
:math:`g_{kl} = -\frac{1}{2} \partial_{\theta'_{k}} \partial_{\theta'_{l}} F |_{\boldsymbol{\theta}' = \boldsymbol{\theta}}`.
The overlap is the probability of measuring all qubits in :math:`\ket{0}` after running the circuit
at :math:`\boldsymbol{\theta}` followed by the inverse of the circuit at
:math:`\boldsymbol{\theta}'`, which is estimated with ``metric_nshots`` shots.
For :math:`P` rotations, this requires :math:`\mathcal{O}(P^{2})` circuits of twice the depth,
and the transpiler, the noise model and the other settings of the ``decoding`` are used to run
them, as it happens for the loss.

The statistical error of each entry of the metric tensor decreases as
:math:`1 / \sqrt{\texttt{metric\_nshots}}`, therefore ``regularization`` should be larger than the
expected error, and the estimated metric tensor is symmetric but not guaranteed to be
positive semi-definite. Only the metric tensor is affected by ``metric_nshots``. To sample the
gradient as well, set ``nshots`` in the :class:`qiboml.models.decoding.Expectation` decoding and
use the default :class:`qiboml.differentiations.psr.PSR` engine, which also requires the observable
to be a symbolic Hamiltonian, as ``qibo`` cannot yet estimate the expectation value of other
non-diagonal observables from samples:

.. code-block:: python

    decoding = Expectation(
        nqubits=nqubits,
        observable=TFIM(nqubits, h=1.0, dense=False, backend=backend),
        nshots=10000,
        backend=backend,
    )
    optimizer = QuantumNaturalGradient(
        circuit, decoding, metric_nshots=10000, regularization=1e-1
    )

Example usage - VQE for the transverse-field Ising model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here we use the QNG optimizer to run the Variational Quantum Eigensolver (VQE) for the ground state
of the transverse-field Ising model on three qubits, with an ansatz containing
:class:`qibo.gates.U3` and :class:`qibo.gates.RBS` gates.

.. code-block:: python

    import math

    from qibo import Circuit, gates, get_backend
    from qibo.hamiltonians import TFIM
    from qiboml.differentiations import Adjoint
    from qiboml.models.decoding import Expectation
    from qiboml.models.optimizers import QuantumNaturalGradient

    backend = get_backend()

    nqubits = 3
    hamiltonian = TFIM(nqubits, h=1.0, backend=backend)
    decoding = Expectation(nqubits=nqubits, observable=hamiltonian, backend=backend)

    circuit = Circuit(nqubits)
    for qubit in range(nqubits):
        circuit.add(gates.U3(qubit, *backend.random_uniform(0, 2 * math.pi, 3)))
    circuit.add(gates.RBS(0, 1, backend.random_uniform(0, 2 * math.pi)))
    circuit.add(gates.RBS(1, 2, backend.random_uniform(0, 2 * math.pi)))
    for qubit in range(nqubits):
        circuit.add(gates.U3(qubit, *backend.random_uniform(0, 2 * math.pi, 3)))

    optimizer = QuantumNaturalGradient(
        circuit,
        decoding,
        differentiation=Adjoint,
        learning_rate=0.1,
        regularization=1e-2,
    )
    final_loss, losses, final_parameters = optimizer(steps=100)

When constructing a :class:`qiboml.models.optimizers.QuantumNaturalGradient` object,
the arguments are:

- ``circuit``: Circuit to be optimized.
- ``decoding``: :class:`qiboml.models.decoding.Expectation` decoding defining the loss.
- ``differentiation``: Differentiation engine class. Default
    :class:`qiboml.differentiations.psr.PSR`.
- ``metric_nshots``: Number of shots used to estimate each overlap needed by the metric tensor.
    If ``None``, the metric tensor is computed exactly. Default ``None``.
- ``learning_rate``: Learning rate :math:`\eta`. Default :math:`0.1`.
- ``regularization``: Regularization strength :math:`\lambda`. Default :math:`10^{-2}`.
- ``callback``: Optional callback function, called at every iteration with the keyword arguments
    ``iter_num``, ``loss`` and ``parameters``.

When calling the optimizer, ``steps`` sets the maximum number of iterations, and the optimization
stops earlier if the norm of the update direction :math:`(g + \lambda I)^{-1} \nabla \mathcal{L}`
falls below ``tolerance``. At the end of the run, the following objects are returned:

- ``final_loss``: Loss at final parameters.
- ``losses``: Loss at each iteration.
- ``final_parameters``: Final parameters.
