import math
from typing import Any, Callable, Optional, Tuple

from numpy.typing import ArrayLike
from qibo import Circuit, gates
from qibo.backends import Backend, HammingWeightBackend, _check_backend
from qibo.config import log, raise_error
from qibo.models._encodings import _ehrlich_algorithm, _generate_rbs_angles
from qibo.models.encodings import hamming_weight_encoder
from qibo.quantum_info import random_statevector
from scipy.sparse import issparse, isspmatrix_coo
from scipy.special import comb

from qiboml.differentiations.abstract import Differentiation
from qiboml.differentiations.psr import PSR
from qiboml.models.decoding import Expectation, Samples, State


class ExactGeodesicTransportCG:
    """Exact Geodesic Transport with Conjugate Gradients Optimizer.

    Implements the Exact Geodesic Transport with Conjugate Gradients (EGT-CG) optimizer,
    a curvature-aware Riemannian optimizer designed specifically for variational circuits based
    on the Hamming-weight encoder (HWE) ansatze. It updates parameters along exact geodesic
    paths on the hyperspherical manifold defined by the HWE, combining analytic metric
    computation, conjugate-gradient memory, and dynamic learning rates for fast, globally
    convergent optimization.

    Args:
        nqubits (int): Number of qubits in the quantum circuit.
        weight (int): Hamming weight to encode.
        loss_fn (str or Callable): if str, only possibility is ``exp_val``,
            for expectation value loss (i.e., running VQE).
            It can also be a Callable to be used as the loss function.
            First two arguments (mandatory) are circuit and backend for execution.
        loss_kwargs: (dict, optional): Additional arguments to be passed to the loss function.
            For VQE (``loss_fn = "exp_val"``), include item ``"hamiltonian": hamiltonian``,
            where ``hamiltonian`` is passed as ``ArrayLike``, scipy sparse or
            backend-specific sparse.
        initial_parameters (ArrayLike, optional): Initial hyperspherical angles parameterizing
            the amplitudes. If None, initializes from a Haar-random state.
        backtrack_rate (float, optional): Backtracking rate for Wolfe condition
            line search. Defaults to :math:`0.9`.
        backtrack_multiplier (float, optional): Scaling factor applied to the initial learning
            rate for the backtrack. Usually, it's greater than 1 to guarantee a wider
            search space. Defaults to :math:`1.5`.
        backtrack_min_lr (float, optional): Minimum learning rate to be tested in the backtrack.
            Defaults to :math:`10^{-6}`.
        c1 (float, optional): Constant for Armijo condition (sufficient decrease) in
            Wolfe line search. Defaults to :math:`10^{-3}`.
        c2 (float, optional): Constant for curvature condition in Wolfe line search.
            It should satisfy ``c1 < c2 < 1``. Defaults to :math:`0.9`.
        callback (Callable, optinal): callback function. First two positional arguments are
            ``iter_number`` and ``loss_value``.
        seed (int, optional): random seed. Controls initialization.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be used
            in the execution. If ``None``, it uses the current backend.
            Defaults to ``None``.

    Returns:
        :class:`qiboml.models.optimizers.ExactGeodesicTransportCG`: Instantiated optimizer object.

    References:
        A. J. Ferreira-Martins, R. M. S. Farias, G. Camilo, T. O. Maciel, A. Tosta,
        R. Lin, A. Alhajri, T. Haug, and L. Aolita, *Quantum optimization
        with exact geodesic transport*, `arXiv:2506.17395 (2025)
        <https://arxiv.org/abs/2506.17395>`_.
    """

    def __init__(
        self,
        nqubits: int,
        weight: int,
        loss_fn: Callable[..., tuple[float, Any]],
        loss_kwargs: dict | None = None,
        initial_parameters: ArrayLike | None = None,
        backtrack_rate: float = 0.9,
        backtrack_multiplier: float = 1.5,
        backtrack_min_lr: float = 1e-6,
        c1: float = 0.001,
        c2: float = 0.9,
        callback: Callable[..., None] | None = None,
        seed: int | None = None,
        backend: Backend = None,
    ):
        self.nqubits = nqubits
        self.weight = weight
        self.backtrack_rate = backtrack_rate
        self.backtrack_multiplier = backtrack_multiplier
        self.backtrack_min_lr = backtrack_min_lr
        self.c1 = c1
        self.c2 = c2
        if backend is None:  # pragma: no cover
            backend = HammingWeightBackend("numpy")
        self.backend = _check_backend(backend)
        self.callback = callback
        self.n_calls_loss = 0
        self.n_calls_gradient = 0
        self.loss_fn = loss_fn

        if initial_parameters is not None:
            dtype = (
                initial_parameters[0].dtype
                if isinstance(initial_parameters, list)
                else initial_parameters.dtype
            )
            self.angles = self.backend.cast(initial_parameters, dtype=dtype)
        else:
            self.angles = random_statevector(
                int(comb(nqubits, weight)),
                seed=seed,
                backend=self.backend,
                dtype=self.backend.float64,
            )
            self.angles = _generate_rbs_angles(
                self.angles,
                architecture="diagonal",
                backend=self.backend,
            )
            self.angles = self.backend.cast(self.angles, dtype=self.angles[0].dtype)

        self.x = self.angles_to_amplitudes(self.angles)
        self.circuit = hamming_weight_encoder(
            nqubits=self.nqubits,
            weight=self.weight,
            data=self.x,
            full_hwp=bool(self.backend.name == "hamming_weight"),
            backend=self.backend,
        )
        self.angles = self.backend.cast(
            [x[0] for x in self.circuit.get_parameters()],
            dtype=self.backend.float64,
        )

        self.riemannian_tangent = False

        if not isinstance(loss_fn, (str, Callable)):
            raise_error(
                TypeError,
                "``loss_fn`` must be either the str ``exp_val`` or "
                + f"``Callable``. Passed {type(loss_fn)}",
            )
        elif isinstance(loss_fn, str) and loss_fn != "exp_val":
            raise_error(
                ValueError,
                f"If str, ``loss_fn`` can only be ``exp_val``. Passed {type(loss_fn)}.",
            )

        self.hamiltonian = None
        if "hamiltonian" in loss_kwargs:
            self.hamiltonian = loss_kwargs.get("hamiltonian", None)
            if not (
                isinstance(self.hamiltonian, self.backend.tensor_types)
                or issparse(self.hamiltonian)
            ):
                raise_error(
                    TypeError,
                    f"For backend <{self.backend}>, ``hamiltonian`` must be scipy sparse matrix, "
                    + f"or one of these: {self.backend.tensor_types}\n"
                    + f"passed type: {type(self.hamiltonian)}!",
                )
            if issparse(self.hamiltonian):
                if (
                    self.backend.platform is not None
                    and self.backend.platform != "numpy"
                ):
                    self.hamiltonian = _scipy_sparse_to_backend_coo(
                        self.hamiltonian, self.backend
                    )
            else:
                if (
                    self.backend.platform is not None
                    and self.backend.platform != "numpy"
                ):
                    self.hamiltonian = self.backend.coo_matrix(self.hamiltonian)
                else:
                    self.hamiltonian = self.backend.csr_matrix(self.hamiltonian)

            loss_kwargs["hamiltonian"] = self.hamiltonian

        backends_autodiff = ["jax", "tensorflow", "pytorch"]
        if loss_fn == "exp_val" or self.backend.name == "hamming_weight":
            if self.hamiltonian is None:
                raise_error(
                    ValueError,
                    "For ``loss_fn='exp_val'``, you must pass the hamiltonian to ``loss_kwargs`` "
                    + "via the dict item ``{'hamiltonian': hamiltonian}``.",
                )

            if self.backend.platform in backends_autodiff:
                self.hamiltonian_subspace = None
                self.loss_fn = _loss_func_expval
                self.gradient_func = self._gradient_func_internal
            else:
                self.hamiltonian_subspace = self.get_subspace_hamiltonian()
                if self.backend.name == "hamming_weight":  # pragma: no cover
                    loss_kwargs["hamiltonian"] = self.hamiltonian_subspace
                self.loss_fn = _loss_func_expval
                self.riemannian_tangent = True
                self.gradient_func = None
        else:
            if self.backend.platform not in backends_autodiff:
                raise_error(
                    TypeError,
                    f"To use autodiff, must use one of the following backends: {backends_autodiff}",
                )
            self.hamiltonian_subspace = None
            self.gradient_func = self._gradient_func_internal

        self.loss = self._loss_internal
        self.loss_kwargs = loss_kwargs
        if self.backend.name == "hamming_weight":  # pragma: no cover
            self.loss_kwargs["weight"] = self.weight

        self.jacobian = None
        self.inverse_jacobian = None

        initial_string = initial_string = [1] * self.weight + [0] * (
            self.nqubits - self.weight
        )
        bitstrings_ehrlich = _ehrlich_algorithm(initial_string, False)
        bitstrings_lex = sorted(bitstrings_ehrlich)
        self.reindex_list = [bitstrings_ehrlich.index(bs) for bs in bitstrings_lex]
        if self.backend.platform == "jax":
            self.reindex_list = self.backend.cast(
                self.reindex_list, dtype=self.backend.int32
            )

    def get_subspace_hamiltonian(self) -> ArrayLike:
        """Computes the Hamiltonian restricted to the fixed-weight subspace
        and represented as a dense matrix in the active backend.

        Assumes ``self.hamiltonian`` is in COO format of the respective backend.

        Returns:
            ArrayLike: Dense matrix representing the hamiltonian in the subspace.
        """

        subspace_dim = int(comb(self.nqubits, self.weight))
        initial_string = [1] * self.weight + [0] * (self.nqubits - self.weight)
        lexicographical_order = _ehrlich_algorithm(initial_string, False)
        lexicographical_order.sort()
        basis_states_subspace = [
            int(bitstring, 2) for bitstring in lexicographical_order
        ]
        full_to_sub = {full: i for i, full in enumerate(basis_states_subspace)}

        hamilt_subspace = self.backend.zeros(
            (subspace_dim, subspace_dim), dtype=self.x.dtype
        )

        platform = self.backend.platform
        if platform == "jax":  # pragma: no cover
            indices = self.hamiltonian.indices
            data = self.hamiltonian.data
            rows = indices[:, 0]
            cols = indices[:, 1]
        elif platform == "tensorflow":  # pragma: no cover
            indices = self.hamiltonian.indices.numpy()
            data = self.hamiltonian.values.numpy()
            rows = indices[:, 0]
            cols = indices[:, 1]
        elif platform == "pytorch":  # pragma: no cover
            hamilt = self.hamiltonian.coalesce()
            indices = hamilt.indices()
            values = hamilt.values()
            rows = indices[0].cpu().numpy()
            cols = indices[1].cpu().numpy()
            data = values.cpu().numpy()
        else:
            hamilt = self.hamiltonian.tocoo()
            rows = hamilt.row
            cols = hamilt.col
            data = hamilt.data

        tol = 1e-14
        for i_full, j_full, v in zip(rows, cols, data):
            if abs(v) <= tol:
                continue  # pragma: no cover
            i = full_to_sub.get(int(i_full))
            j = full_to_sub.get(int(j_full))
            if i is None or j is None:
                continue
            if platform == "jax":  # pragma: no cover
                hamilt_subspace = hamilt_subspace.at[(i, j), (j, i)].set(v)
            elif platform == "tensorflow":  # pragma: no cover
                v = self.backend.cast(v, hamilt_subspace.dtype)
                indices = self.backend.engine.constant(
                    [[i, j], [j, i]],
                    dtype=self.backend.int32,
                )
                updates = self.backend.engine.stack([v, v])
                hamilt_subspace = self.backend.engine.tensor_scatter_nd_update(
                    hamilt_subspace, indices, updates
                )
            else:
                hamilt_subspace[i, j] = v
                hamilt_subspace[j, i] = v
        return hamilt_subspace

    def initialize_cg_state(self):
        """Initialize CG state.

        Sets up the internal variables ``x``, ``u``, ``v``, and initial step size ``eta``
        based on the current angles.
        """
        self.v = self.tangent_vector()
        self.u = self.backend.cast(self.v, dtype=self.v.dtype, copy=True)
        norm_u = self.backend.vector_norm(self.u)
        loss_prev = self.loss(self.circuit, self.backend, **self.loss_kwargs)
        self.eta = (1 / norm_u) * self.backend.arccos(
            (1 + (norm_u / (2 * loss_prev)) ** 2) ** -0.5
        )

    def angles_to_amplitudes(self, angles: ArrayLike) -> ArrayLike:
        """Convert angles to amplitudes.

        Args:
           angles (ArrayLike): Angles in hyperspherical coordinates.

        Returns:
            ArrayLike: Amplitudes calculated from the hyperspherical coordinates.
        """
        d = len(angles) + 1
        amps = []
        for k in range(d):
            prod = self.backend.prod(self.backend.sin(angles[:k]))
            if k < d - 1:
                prod *= self.backend.cos(angles[k])
            amps.append(prod)
        amps = self.backend.cast(amps, dtype=self.backend.float64)
        return amps

    def get_jacobian(self) -> ArrayLike:
        """Compute Jacobian of amplitudes wrt angles.

        Returns:
            ArrayLike: Jacobian matrix.
        """
        dim = len(self.angles)
        jacob = self.backend.zeros((dim + 1, dim), dtype=self.backend.float64)

        for j in range(dim):
            reduced_params = self.backend.cast(
                self.angles[j:], dtype=self.backend.float64, copy=True
            )
            if self.backend.platform == "tensorflow":
                reduced_params = self.backend.engine.tensor_scatter_nd_update(
                    reduced_params, [[0]], [reduced_params[0] + math.pi / 2]
                )
            elif self.backend.platform == "jax":
                reduced_params = reduced_params.at[0].set(
                    reduced_params[0] + math.pi / 2
                )
            else:
                reduced_params[0] += math.pi / 2

            sins = self.backend.prod(self.backend.sin(self.angles[:j]))
            amps = self.angles_to_amplitudes(reduced_params)

            updates = self.backend.real(sins * amps)

            if self.backend.platform == "tensorflow":
                indices = list(range(j, jacob.shape[0]))
                indices = list(zip(indices, [j] * len(indices)))
                jacob = self.backend.engine.tensor_scatter_nd_update(
                    jacob, indices, updates
                )
            elif self.backend.platform == "jax":
                jacob = jacob.at[j:, j].set(updates)
            else:
                jacob[j:, j] = updates

        return jacob

    def metric_tensor(self) -> ArrayLike:
        """Compute the diagonal metric tensor in hyperspherical coordinates.

        Returns:
            ArrayLike: Diagonal elements of the metric tensor.
        """
        g_diag = [
            self.backend.prod(self.backend.sin(self.angles[:k]) ** 2)
            for k in range(len(self.angles))
        ]
        return self.backend.cast(g_diag, dtype="float64")

    def tangent_vector(self) -> ArrayLike:
        """Compute the Riemannian gradient (tangent vector) at the current point on the hypersphere.

        If loss is expectation value, uses the analytical gradient computation from amplitudes.

        If it is a generic loss, performs backpropagation in parameters space, then uses
        the jacobian to go to amplitudes coordinates.

        Returns:
            ArrayLike: Tangent vector in the tangent space of the hypersphere.
        """

        if self.riemannian_tangent:
            l_psi = self.loss(self.circuit, self.backend, **self.loss_kwargs)
            psi_amps = self.x
            self.n_calls_gradient += 1
            return self.backend.real(
                2 * (l_psi * psi_amps - self.hamiltonian_subspace @ psi_amps)
            )

        self.grad = self.gradient_func()
        inv_g = 1.0 / self.metric_tensor()
        nat_grad = -inv_g * self.grad
        self.jacobian = self.get_jacobian()
        return (self.jacobian @ nat_grad)[self.reindex_list]

    def regularization(self, angles: ArrayLike) -> ArrayLike:
        """Applies regularization to vector of parameters after update,
        effectively changing charts away from singularities.
        Returns corresponding amplitudes directly.

        Args:
            angles ArrayLike: vector of parameters.
        Returns:
            ArrayLike:vector of amplitudes post-regularization.
        """
        condition = self.backend.abs(self.backend.sin(angles[:-1])) < 1e-3
        updated = self.backend.where(condition, math.pi / 2, angles[:-1])
        return self.angles_to_amplitudes(
            self.backend.concatenate([updated, angles[-1:]], axis=0)
        )

    def optimize_step_size(
        self, x_prev: ArrayLike, u_prev: ArrayLike, v_prev: ArrayLike, loss_prev: float
    ) -> Tuple[ArrayLike, ArrayLike, float]:
        """Perform Wolfe line search to determine optimal step size eta via the satisfaction
        of the Wolfe conditions.

        Args:
            x_prev (ArrayLike): Previous amplitudes on the sphere.
            u_prev (ArrayLike): Previous conjugate search direction.
            v_prev (ArrayLike): Previous search direction.
            loss_prev (float): Loss at previous amplitudes.

        Returns:
            Tuple: Respectively: updated amplitudes, new search direction, and optimal step size.
        """

        eta = self.backtrack_multiplier * self.eta

        angles_orig = self.angles
        amps_orig = self.x

        while eta > self.backtrack_min_lr:

            transported_u = self.parallel_transport(u_prev, u_prev, x_prev, eta)

            x_new = self.regularization(
                self.amplitudes_to_angles(
                    self.exponential_map_with_direction(u_prev, eta)
                )
            )
            self.circuit = hamming_weight_encoder(
                nqubits=self.nqubits,
                weight=self.weight,
                data=x_new,
                full_hwp=bool(self.backend.name == "hamming_weight"),
                backend=self.backend,
            )
            self.angles = self.backend.cast(
                [x[0] for x in self.circuit.get_parameters()],
                dtype=self.backend.float64,
            )
            self.x = x_new
            loss_new = self.loss(self.circuit, self.backend, **self.loss_kwargs)

            condition_a_lhs = loss_new - loss_prev
            condition_a_rhs = self.c1 * eta * (-v_prev @ u_prev)
            condition_a = condition_a_lhs <= condition_a_rhs

            if condition_a:

                v_new = self.tangent_vector()

                condition_b_lhs = abs(-v_new @ transported_u)
                condition_b_rhs = abs(self.c2 * (-v_prev @ u_prev))
                condition_b = condition_b_lhs <= condition_b_rhs

                if condition_a and condition_b:

                    self.x = amps_orig
                    return x_new, v_new, eta

            eta *= self.backtrack_rate

            self.angles = angles_orig
            self.x = amps_orig

        return x_new, v_new, eta  # pragma: no cover

    def exponential_map_with_direction(
        self,
        direction: ArrayLike,
        eta: float,
    ) -> ArrayLike:
        """Applies xponential map from current point along specified direction.

        Args:
            direction (ArrayLike): Tangent vector direction.
            eta (float): Step size.

        Returns:
            ArrayLike: Amplitudes of new point on the hypersphere.
        """
        norm_dir = self.backend.vector_norm(direction)
        x_new = self.backend.cos(eta * norm_dir) * self.x + self.backend.sin(
            eta * norm_dir
        ) * (direction / norm_dir)
        return x_new

    def amplitudes_to_angles(self, x: ArrayLike) -> ArrayLike:
        """Computes the angles corresponding to a given amplitudes vector.

        Args:
            x (ArrayLike): Amplitudes vector.

        Returns:
            ArrayLike: Corresponding angles.
        """
        d = len(x)
        angles = self.backend.zeros(d - 1, dtype=self.backend.float64)
        for elem in range(d - 2):
            norm_tail = self.backend.vector_norm(x[elem:])
            updates = (
                0.0 if norm_tail == 0 else self.backend.arccos(x[elem] / norm_tail)
            )
            if self.backend.platform == "tensorflow":
                angles = self.backend.engine.tensor_scatter_nd_update(
                    angles, [[elem]], [updates]
                )
            elif self.backend.platform == "jax":
                angles = angles.at[elem].set(updates)
            else:
                angles[elem] = updates

        update = self.backend.arctan2(x[-1], x[-2])
        if self.backend.platform == "tensorflow":
            angles = self.backend.engine.tensor_scatter_nd_update(
                angles, [[len(angles) - 1]], [update]
            )
        elif self.backend.platform == "jax":
            angles = angles.at[-1].set(update)
        else:
            angles[-1] = update

        return angles

    def parallel_transport(
        self, u: ArrayLike, v: ArrayLike, a: ArrayLike, eta=None
    ) -> ArrayLike:
        """Parallel transport a tangent vector u along geodesic defined by v.

        Args:
            u (ArrayLike): Vector to transport.
            v (ArrayLike): Direction of geodesic.
            a (ArrayLike): Starting point on sphere.
            eta (float, optional): Step size. If ``None``, defaults to current ``eta``.
                Defaults to ``None``.

        Returns:
            ArrayLike: Transported vector.
        """
        if eta is None:
            eta = self.eta
        norm_v = self.backend.vector_norm(v)
        vu_dot = v @ u
        transported = (
            u
            - self.backend.sin(eta * norm_v) * (vu_dot / norm_v) * a
            + (self.backend.cos(eta * norm_v) - 1) * (vu_dot / (norm_v**2)) * v
        )
        return transported

    def beta_dy(self, v_next: ArrayLike, transported_u: ArrayLike, st: float) -> float:
        """Compute Dai and Yuan Beta.

        Args:
            v_next (ArrayLike): Next gradient.
            x_next (ArrayLike): Next point.
            transported_u (ArrayLike): Parallel-transported u.
            st (float): Scaling factor.

        Returns:
            float: Dai-Yuan beta value.
        """
        numerator = -v_next @ -v_next
        denominator = (-v_next @ (st * transported_u)) - (-self.v @ self.u)
        return numerator / denominator

    def beta_hs(
        self,
        v_next: ArrayLike,
        transported_u: ArrayLike,
        transported_v: ArrayLike,
        lt: float,
        st: float,
    ) -> float:
        """Compute Hestenes-Stiefel conjugate gradient beta.

        Args:
            v_next (ArrayLike): Next gradient.
            x_next (ArrayLike): Next point.
            transported_u (ArrayLike): Parallel-transported u.
            transported_v (ArrayLike): Parallel-transported v.
            lt (float): Scaling factor.
            st (float): Scaling factor.

        Returns:
            float: Hestenes-Stiefel beta value.
        """
        numerator = (-v_next @ -v_next) - (-v_next @ (lt * transported_v))
        denominator = (-v_next @ (st * transported_u)) - (-self.v @ self.u)
        return numerator / denominator

    def run_egt_cg(
        self, steps: int = 100, tolerance: float = 1e-8
    ) -> Tuple[float, ArrayLike, ArrayLike]:
        """Run the EGT-CG optimizer for a specified number of steps.

        Args:
            steps (int, optional): Number of optimization iterations. Defaults to :math:`100`.
            tolerance (float, optional): Maximum tolerance for the residue of the gradient update.
                Defaults to :math:`10^{-8}`.

        Returns:
            Tuple[float, ArrayLike, ArrayLike]:
                final_loss: Final loss value.
                losses: Loss at each iteration.
                final_parameters: Final optimized parameters (angles).
        """
        self.initialize_cg_state()
        losses = []
        for iter_num in range(steps):
            loss_prev = self.loss(self.circuit, self.backend, **self.loss_kwargs)
            losses.append(loss_prev)

            norm_u = self.backend.vector_norm(self.u)

            res = ((-self.v @ self.u) ** 2) / norm_u
            if res < tolerance:
                print(f"\nOptimized converged at iteration {iter_num+1}!\n")
                break

            if self.callback is not None:
                self.callback(iter_num=iter_num + 1, loss=loss_prev, x=self.x)

            x_prev = self.backend.cast(self.x, dtype=self.x.dtype, copy=True)
            u_prev = self.backend.cast(self.u, dtype=self.u.dtype, copy=True)

            self.eta = (1 / norm_u) * self.backend.arccos(
                (1 + (norm_u / (2 * loss_prev)) ** 2) ** -0.5
            )

            x_new, v_new, new_eta = self.optimize_step_size(
                x_prev=x_prev, u_prev=u_prev, v_prev=self.v, loss_prev=loss_prev
            )
            transported_u = self.parallel_transport(self.u, self.u, self.x)

            st = min(
                1,
                self.backend.sqrt(self.u @ self.u)
                / self.backend.sqrt(transported_u @ transported_u),
            )
            transported_v = self.parallel_transport(self.u, -self.v, self.x)
            lt = min(
                1,
                self.backend.sqrt(self.v @ self.v)
                / self.backend.sqrt(transported_v @ transported_v),
            )
            beta_dy = self.beta_dy(v_next=v_new, transported_u=transported_u, st=st)
            beta_hs = self.beta_hs(
                v_next=v_new,
                transported_u=transported_u,
                transported_v=transported_v,
                lt=lt,
                st=st,
            )

            beta_val = max(0, min(beta_dy, beta_hs))

            self.x = x_new
            self.v = v_new
            self.eta = new_eta
            self.u = v_new + beta_val * st * transported_u

            self.circuit = hamming_weight_encoder(
                nqubits=self.nqubits,
                weight=self.weight,
                data=self.x,
                full_hwp=bool(self.backend.name == "hamming_weight"),
                backend=self.backend,
            )
            self.angles = self.backend.cast(
                [x[0] for x in self.circuit.get_parameters()],
                dtype=self.backend.float64,
            )

        final_loss = self.loss(self.circuit, self.backend, **self.loss_kwargs)
        losses.append(final_loss)
        final_parameters = self.angles
        return (
            final_loss,
            self.backend.cast(losses),
            self.backend.cast(final_parameters, dtype=final_parameters.dtype),
        )

    def __call__(
        self, steps: int = 10, tolerance: float = 1e-8
    ) -> Tuple[float, ArrayLike, ArrayLike]:
        """Run the EGT-CG optimizer for a specified number of steps.

        Args:
            steps (int, optional): Number of optimization iterations. Defaults to :math:`100`.
            tolerance (float, optional): Maximum tolerance for the residue of the gradient update.
                Defaults to :math:`10^{-8}`.

        Returns:
            Tuple[float, ArrayLike, ArrayLike]:
                final_loss: Final loss value.
                losses: Loss at each iteration.
                final_parameters: Final optimized parameters (angles).
        """
        return self.run_egt_cg(steps=steps, tolerance=tolerance)

    def _loss_internal(self, circuit: Circuit, backend: Backend, **kwargs) -> float:
        """Wrapper function for the loss, used to update attribute ``n_call_loss``
        every time the loss is executed.

        Args:
            circuit (:class:`qibo.models.circuit.Circuit`): circuit used to compute the loss.
            backend (:class:`qibo.backends.abstract.Backend`): backend for execution.

        Returns:
            float: value of loss function.
        """
        self.n_calls_loss += 1
        return self.loss_fn(circuit, backend, **kwargs)

    def _gradient_func_internal(self) -> ArrayLike:
        """
        Compute the gradient of ``self.loss`` w.r.t the trainable parameters
        stored inside self.circuit, using backpropagation of the backend specified by ``platform``.
        This is used if loss != ``exp_val``.

        Returns:
            ArrayLike: gradient vector as an array of the backend platform.
        """
        self.n_calls_gradient += 1

        platform = self.backend.platform
        if platform == "jax":
            circuit_orig = self.circuit.copy(deep=True)

            def loss_fn(params):
                self.circuit.set_parameters(params)
                return self.loss(self.circuit, self.backend, **self.loss_kwargs)

            params = self.backend.cast(
                self.circuit.get_parameters(),
                dtype=self.backend.float64,
            )
            grad = self.backend.jax.grad(loss_fn)(params)
            self.circuit = circuit_orig.copy(deep=True)
            return grad.reshape(-1)
        if platform == "pytorch":
            params = self.backend.cast(
                self.circuit.get_parameters(), dtype=self.angles.dtype
            )
            params.requires_grad = True
            self.circuit.set_parameters(params)
            loss = self.loss(self.circuit, self.backend, **self.loss_kwargs)
            loss.backward()
            return params.grad.reshape(-1)
        if platform == "tensorflow":
            params = self.backend.engine.Variable(
                self.circuit.get_parameters(),
                dtype=self.backend.float64,
            )
            with self.backend.engine.GradientTape() as tape:
                self.circuit.set_parameters(params)
                loss = self.loss(self.circuit, self.backend, **self.loss_kwargs)
            grad = tape.gradient(loss, params)
            return grad.reshape(-1)


class QuantumNaturalGradient:
    """Quantum natural gradient (QNG) optimizer.

    Updates the trainable parameters :math:`\\boldsymbol{\\theta}` of a circuit with the rule
    :math:`\\boldsymbol{\\theta} \\leftarrow \\boldsymbol{\\theta} - \\eta \\,
    (g + \\lambda I)^{-1} \\, \\nabla \\mathcal{L}`, where :math:`\\eta` is the learning rate,
    :math:`\\lambda` is a regularization strength, :math:`I` is the identity matrix,
    :math:`\\nabla \\mathcal{L}` is the gradient of the loss and :math:`g` is the
    Fubini-Study metric tensor of the circuit state :math:`\\ket{\\psi}`, with entries
    :math:`g_{kl} = \\mathrm{Re}\\left[\\braket{\\partial_{k} \\psi | \\partial_{l} \\psi}
    - \\braket{\\partial_{k} \\psi | \\psi} \\braket{\\psi | \\partial_{l} \\psi}\\right]`,
    where :math:`\\partial_{k}` is the derivative with respect to the :math:`k`-th parameter.

    The circuit is internally decomposed into fixed gates and single-parameter Pauli rotations,
    whose angles are linear functions of the parameters of the original gates, e.g.
    :math:`U_{3}(\\theta, \\phi, \\lambda) = R_{Z}(\\phi) R_{Y}(\\theta) R_{Z}(\\lambda)`,
    where :math:`R_{Z}` and :math:`R_{Y}` are rotations around the :math:`Z` and :math:`Y` axes.
    Derivatives are taken with respect to the angles of the rotations and mapped back to
    the parameters of the circuit with the chain rule. The trainable gates that are supported
    are :class:`qibo.gates.RX`, :class:`qibo.gates.RY`, :class:`qibo.gates.RZ`,
    :class:`qibo.gates.U1`, :class:`qibo.gates.U2`, :class:`qibo.gates.U3` and all the gates
    whose ``decompose`` method returns such rotations, e.g. :class:`qibo.gates.RBS`,
    :class:`qibo.gates.GIVENS`, :class:`qibo.gates.RXX`, :class:`qibo.gates.RYY`,
    :class:`qibo.gates.RZZ`, :class:`qibo.gates.RZX`, :class:`qibo.gates.CRX`,
    :class:`qibo.gates.CRY` and :class:`qibo.gates.CRZ`.

    Args:
        circuit (:class:`qibo.models.circuit.Circuit`): circuit to be optimized.
        decoding (:class:`qiboml.models.decoding.Expectation`): decoding that defines the loss,
            i.e. the expectation value of its observable. It also sets the backend
            used by the optimizer.
        differentiation (type[:class:`qiboml.differentiations.abstract.Differentiation`], optional):
            differentiation engine used to compute the gradient of the loss, e.g.
            :class:`qiboml.differentiations.psr.PSR`,
            :class:`qiboml.differentiations.adjoint.Adjoint` or
            :class:`qiboml.differentiations.jax.Jax`. If it is a
            :class:`qiboml.differentiations.jax.Jax` engine, it also computes the derivatives
            of the state needed by the metric tensor. Otherwise, these are computed with the
            parameter-shift rule applied to the state vector. Defaults to
            :class:`qiboml.differentiations.psr.PSR`.
        metric_nshots (int, optional): number of shots used to estimate each overlap needed
            by the metric tensor. If ``None``, the metric tensor is computed exactly from the
            state vector. Otherwise, it is estimated by sampling, with the parameter-shift rule
            applied to the overlap :math:`|\\braket{\\psi(\\boldsymbol{\\theta}) |
            \\psi(\\boldsymbol{\\theta}')}|^{2}` between the states prepared by the circuit at
            two sets of parameters, which is the probability of measuring all qubits in
            :math:`\\ket{0}` after running the circuit at :math:`\\boldsymbol{\\theta}'`
            inverted, after the circuit at :math:`\\boldsymbol{\\theta}`. This costs
            :math:`\\mathcal{O}(P^{2})` circuits for :math:`P` rotations, and the sampling
            uses the transpiler, noise model and the other settings of ``decoding``.
            The statistical error of the estimated metric tensor decreases as
            :math:`1 / \\sqrt{\\mathtt{metric\\_nshots}}`, so ``regularization`` should be
            larger than that. Defaults to ``None``.
        learning_rate (float, optional): learning rate :math:`\\eta`. Defaults to :math:`0.1`.
        regularization (float, optional): strength :math:`\\lambda` of the diagonal shift added
            to the metric tensor before inversion, which stabilizes the update when the
            metric is singular. Defaults to :math:`10^{-2}`.
        callback (Callable, optional): callback function. Keyword arguments are
            ``iter_num``, ``loss`` and ``parameters``. Defaults to ``None``.

    References:
        J. Stokes, J. Izaac, N. Killoran, and G. Carleo, *Quantum Natural Gradient*,
        `Quantum 4, 269 (2020) <https://doi.org/10.22331/q-2020-05-25-269>`_.
    """

    def __init__(
        self,
        circuit: Circuit,
        decoding: Expectation,
        differentiation: type[Differentiation] = PSR,
        metric_nshots: int | None = None,
        learning_rate: float = 0.1,
        regularization: float = 1e-2,
        callback: Callable[..., None] | None = None,
    ):
        if not isinstance(decoding, Expectation):
            raise_error(
                TypeError,
                f"``decoding`` must be an ``Expectation``. Passed {type(decoding)}.",
            )
        if not (
            isinstance(differentiation, type)
            and issubclass(differentiation, Differentiation)
        ):
            raise_error(
                TypeError,
                "``differentiation`` must be a ``Differentiation`` class. "
                + f"Passed {differentiation}.",
            )

        if metric_nshots is not None and (
            not isinstance(metric_nshots, int) or metric_nshots < 1
        ):
            raise_error(
                ValueError,
                f"``metric_nshots`` must be a positive integer. Passed {metric_nshots}.",
            )

        self.backend = decoding.backend
        self.circuit = circuit
        self.decoding = decoding
        self.metric_nshots = metric_nshots
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.callback = callback
        self.parameters = self.backend.cast(
            [
                parameter
                for gate in self.circuit.trainable_gates
                for parameter in gate.parameters
            ],
            dtype=self.backend.float64,
        )
        if len(self.parameters) == 0:
            raise_error(ValueError, "The circuit has no trainable parameters.")

        self._decomposed, self._map = self._decompose()
        self._engine = differentiation(circuit=self._decomposed, decoding=decoding)
        self._state_engine = None
        self._samples_kwargs = None
        if metric_nshots is not None:
            self._samples_kwargs = {
                "nqubits": decoding.nqubits,
                "qubits": decoding.qubits,
                "wire_names": decoding.wire_names,
                "nshots": metric_nshots,
                "backend": self.backend,
                "transpiler": decoding.transpiler,
                "noise_model": decoding.noise_model,
                "density_matrix": decoding.density_matrix,
            }
        else:
            try:
                from qiboml.differentiations import Jax  # pylint: disable=C0415

                if issubclass(differentiation, Jax):
                    self._state_engine = differentiation(
                        circuit=self._decomposed,
                        decoding=State(nqubits=decoding.nqubits, backend=self.backend),
                    )
            except ImportError:  # pragma: no cover
                pass

    def __call__(
        self, steps: int = 100, tolerance: float = 1e-8
    ) -> tuple[float, ArrayLike, ArrayLike]:
        """Run the QNG optimizer for a specified number of steps.

        Args:
            steps (int, optional): number of optimization iterations. Defaults to :math:`100`.
            tolerance (float, optional): the optimization stops when the norm of the natural
                gradient :math:`(g + \\lambda I)^{-1} \\nabla \\mathcal{L}` is below this value.
                Defaults to :math:`10^{-8}`.

        Returns:
            tuple[float, ArrayLike, ArrayLike]: final loss, loss at each iteration
            and final parameters.
        """
        losses = []
        for iter_num in range(1, steps + 1):
            loss = self.decoding(self.circuit)[0, 0]
            losses.append(loss)

            if self.callback is not None:
                self.callback(iter_num=iter_num, loss=loss, parameters=self.parameters)

            gradient, metric = self._gradient_and_metric()
            metric = metric + self.regularization * self.backend.real(
                self.backend.matrices.I(len(self.parameters))
            )
            natural_gradient = self.backend.einsum(
                "ij,j->i", self.backend.inv(metric), gradient
            )
            if self.backend.vector_norm(natural_gradient) < tolerance:
                break

            self.parameters = self.parameters - self.learning_rate * natural_gradient
            self.circuit.set_parameters(self.parameters)
        else:
            losses.append(self.decoding(self.circuit)[0, 0])

        return (
            losses[-1],
            self.backend.cast(losses, dtype=self.backend.float64),
            self.parameters,
        )

    def _decompose(self) -> tuple[Circuit, ArrayLike]:
        """Decompose the circuit into fixed gates and single-parameter Pauli rotations.

        Returns:
            tuple[:class:`qibo.models.circuit.Circuit`, ArrayLike]: decomposed circuit and
            matrix that maps the parameters of the circuit to the angles of the rotations.
        """
        decomposed = Circuit(self.circuit.nqubits)
        rows = []
        offset = 0
        for gate in self.circuit.queue:
            if not (isinstance(gate, gates.ParametrizedGate) and gate.trainable):
                decomposed.add(gate)
                continue

            # each piece is a gate, the index of the parameter that its angle depends on
            # (``None`` for a fixed gate) and the proportionality coefficient
            qubit = gate.target_qubits[0]
            if gate.name == "u1":
                pieces = [(gates.RZ(qubit, 0.0), 0, 1.0)]
            elif gate.name == "u2":
                pieces = [
                    (gates.RZ(qubit, 0.0), 1, 1.0),
                    (gates.RY(qubit, math.pi / 2, trainable=False), None, 0.0),
                    (gates.RZ(qubit, 0.0), 0, 1.0),
                ]
            elif gate.name == "u3":
                pieces = [
                    (gates.RZ(qubit, 0.0), 2, 1.0),
                    (gates.RY(qubit, 0.0), 0, 1.0),
                    (gates.RZ(qubit, 0.0), 1, 1.0),
                ]
            else:
                pieces = []
                if len(gate.parameters) != 1:
                    raise_error(
                        NotImplementedError,
                        f"Gate ``{gate.name}`` cannot be decomposed into Pauli rotations.",
                    )
                # a decomposition at two angles tells whether each rotation is fixed
                # or proportional to the angle of the gate
                probes = [
                    gate.__class__(*gate.init_args, angle).decompose()
                    for angle in (1.0, 2.0)
                ]
                for first, second in zip(*probes):
                    if not isinstance(first, gates.ParametrizedGate):
                        pieces.append((first, None, 0.0))
                        continue
                    one, two = first.parameters[0], second.parameters[0]
                    if first.name not in ("rx", "ry", "rz") or not (
                        math.isclose(one, two) or math.isclose(2 * one, two)
                    ):
                        raise_error(
                            NotImplementedError,
                            f"Gate ``{gate.name}`` cannot be decomposed into Pauli rotations.",
                        )
                    if math.isclose(one, two):
                        first = first.__class__(
                            *first.target_qubits, one, trainable=False
                        )
                        pieces.append((first, None, 0.0))
                    else:
                        pieces.append((first, 0, one))

            for piece, index, coefficient in pieces:
                if index is None:
                    decomposed.add(piece)
                    continue
                angle = coefficient * gate.parameters[index]
                decomposed.add(piece.__class__(*piece.target_qubits, angle))
                row = [0.0] * len(self.parameters)
                row[offset + index] = coefficient
                rows.append(row)
            offset += len(gate.parameters)

        return decomposed, self.backend.cast(rows, dtype=self.backend.float64)

    def _gradient_and_metric(self) -> tuple[ArrayLike, ArrayLike]:
        """Compute the loss gradient and the Fubini-Study metric tensor.

        Returns:
            tuple[ArrayLike, ArrayLike]: gradient of the loss and metric tensor.
        """
        angles = self.backend.einsum("ij,j->i", self._map, self.parameters)
        self._decomposed.set_parameters(angles)
        gradient = self.backend.reshape(self._engine.evaluate(angles), (-1,))

        if self._samples_kwargs is not None:
            # g_kl = -1/2 d^2 F / (d theta'_k d theta'_l), with F the overlap between the
            # states at theta and theta', and where the derivatives are evaluated with
            # the parameter-shift rule for functions with a single frequency
            nangles = len(angles)
            reference = self._decomposed.copy(deep=True)
            entries = [[0.0] * nangles for _ in range(nangles)]
            for k in range(nangles):
                for l in range(k, nangles):
                    if k == l:
                        terms = [(0.0, 0.0, 0.25), (math.pi, 0.0, -0.25)]
                    else:
                        terms = [
                            (s * math.pi / 2, t * math.pi / 2, -s * t / 8)
                            for s in (1, -1)
                            for t in (1, -1)
                        ]
                    for shift_k, shift_l, weight in terms:
                        shifted_angles = PSR.shift_parameter(
                            self.backend.cast(angles, copy=True),
                            k,
                            shift_k,
                            self.backend,
                        )
                        shifted_angles = PSR.shift_parameter(
                            shifted_angles, l, shift_l, self.backend
                        )
                        shifted = self._decomposed.copy(deep=True)
                        shifted.set_parameters(shifted_angles)
                        # a new decoding is needed for every circuit, since the measurement
                        # gate of a decoding keeps the samples of its first execution
                        samples = Samples(**self._samples_kwargs)(
                            reference + shifted.invert()
                        )
                        zeros = self.backend.sum(
                            self.backend.cast(
                                self.backend.sum(samples, axis=1) == 0,
                                dtype=self.backend.float64,
                            )
                        )
                        entries[k][l] += weight * float(zeros) / self.metric_nshots
                    entries[l][k] = entries[k][l]
            metric = self.backend.cast(entries, dtype=self.backend.float64)
        else:
            if self._state_engine is None:
                derivatives = []
                for k in range(len(angles)):
                    states = []
                    for sign in (1, -1):
                        self._decomposed.set_parameters(
                            PSR.shift_parameter(
                                self.backend.cast(angles, copy=True),
                                k,
                                sign * math.pi / 2,
                                self.backend,
                            )
                        )
                        states.append(
                            self.backend.execute_circuit(self._decomposed).state()
                        )
                    derivatives.append((states[0] - states[1]) / 2**1.5)
                derivatives = self.backend.cast(derivatives)
            else:
                jacobian = self._state_engine.evaluate(angles)
                derivatives = jacobian[:, 0, 0] + 1j * jacobian[:, 1, 0]

            self._decomposed.set_parameters(angles)
            state = self.backend.execute_circuit(self._decomposed).state()
            projections = self.backend.einsum(
                "ij,j->i", self.backend.conj(derivatives), state
            )
            metric = self.backend.real(
                self.backend.matmul(
                    self.backend.conj(derivatives),
                    self.backend.transpose(derivatives, (1, 0)),
                )
                - self.backend.outer(projections, self.backend.conj(projections))
            )

        map_transpose = self.backend.transpose(self._map, (1, 0))
        return (
            self.backend.einsum(
                "ij,j->i",
                map_transpose,
                self.backend.cast(gradient, dtype=self.backend.float64),
            ),
            self.backend.matmul(map_transpose, self.backend.matmul(metric, self._map)),
        )


def _scipy_sparse_to_backend_coo(matrix, backend: Backend) -> ArrayLike:
    """Convert a SciPy sparse matrix (CSR or COO) to the COO sparse
    representation supported by JAX, TensorFlow, or PyTorch.

    Args:
        matrix (scipy.sparse.csr_matrix or scipy.sparse.coo_matrix): input sparse matrix.
        backend (:class:`qibo.backends.abstract.Backend`): backend used,

    Returns:
        ArrayLike: Backend-specific sparse tensor.
    """

    platform = backend.platform

    if platform == "jax":
        if not isspmatrix_coo(matrix):
            matrix = matrix.tocoo()

        indices = backend.engine.stack([matrix.row, matrix.col], axis=1)
        data = matrix.data

        from jax.experimental.sparse import BCOO  # pylint: disable=C0415

        return BCOO(
            (backend.engine.asarray(data), backend.engine.asarray(indices)),
            shape=matrix.shape,
        )

    if platform == "tensorflow":
        if not isspmatrix_coo(matrix):
            matrix = matrix.tocoo()

        indices = backend.engine.stack([matrix.row, matrix.col], axis=1)
        data = matrix.data

        return backend.engine.sparse.SparseTensor(
            indices=indices.astype(backend.engine.int64),
            values=data,
            dense_shape=matrix.shape,
        )

    if not isspmatrix_coo(matrix):
        matrix = matrix.tocoo()

    row, col = matrix.row, matrix.col
    indices = backend.vstack([backend.cast(row), backend.cast(col)])
    values = matrix.data
    values = backend.cast(values)

    return backend.engine.sparse_coo_tensor(indices, values, size=matrix.shape)


def _loss_func_expval(
    circuit: Circuit,
    backend: Backend,
    *,
    hamiltonian: ArrayLike,
    weight: Optional[int] = None,
) -> float:
    """Backend-agnostic expectation value :math:`\\bra{\\psi} H \\ket{\\psi}`.

    Supports:
    - NumPy / SciPy sparse
    - JAX (BCOO)
    - TensorFlow (tf.sparse.SparseTensor)
    - PyTorch (sparse COO / CSR)

    Assumes Hamiltonian is sparse in the backend's native format

    Args:
        circuit (:class:`qibo.models.circuit.Circuit`): quantum circuit used to compute the loss.
        backend (:class:`qibo.backends.abstract.Backend`): backend for execution.
        hamiltonian (ArrayLike): sparse Hamiltonian in the backend's format.
        weight (int): integer indicating subspace HW, useful for HW backend simulation.

    Returns:
        float: Expectation value.
    """
    kwargs = {"weight": weight} if backend.name == "hamming_weight" else {}
    psi = backend.execute_circuit(circuit, **kwargs).state()
    platform = backend.platform
    if platform == "tensorflow":
        if "cpu" in backend.device.lower():
            psi_col = backend.reshape(psi, (-1, 1))
            h_psi = backend.engine.sparse.sparse_dense_matmul(hamiltonian, psi_col)
            h_psi = backend.reshape(h_psi, (-1,))
        else:  # pragma: no cover
            log.warning(
                "For TensorFlow in GPU, matmul between sparse and dense is not implemented yet. "
                + "Hamiltonian has to be casted to dense for computation."
            )
            psi_col = backend.reshape(psi, (-1, 1))
            h_psi = backend.matmul(backend.engine.sparse.to_dense(hamiltonian), psi_col)
            h_psi = backend.reshape(h_psi, (-1,))

    elif platform == "pytorch":
        h_psi = backend.engine.sparse.mm(hamiltonian, psi.unsqueeze(1)).squeeze(1)
    else:
        h_psi = hamiltonian @ psi

    return backend.real(backend.sum(backend.conj(psi) * h_psi))
