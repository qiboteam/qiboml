import math
from typing import Callable, Any, Tuple

from qibo import Circuit
from qibo.models.encodings import hamming_weight_encoder, _ehrlich_algorithm
from qibo.backends import _check_backend, Backend
from qibo.quantum_info import random_statevector
from qibo.models.encodings import _generate_rbs_angles
from qibo.config import raise_error

from scipy.sparse import issparse, isspmatrix_coo
from scipy.special import comb

from numpy.typing import ArrayLike


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
        loss_fn (str or Callable): if str, only possibility is `exp_val`,
            for expectation value loss (i.e., running VQE).
            It can also be a Callable to be used as the loss function.
            First two arguments (mandatory) are circuit and backend for execution.
        loss_kwargs: (dict, optional): Additional arguments to be passed to the loss function.
            For VQE (`loss_fn = "exp_val"`), include item `"hamiltonian": hamiltonian`,
            where `hamiltonian` is passed as `ArrayLike`, scipy sparse or backend-specific sparse.
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
            `iter_number` and `loss_value`.
        seed (int, optional): random seed. Controls initialization.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be used
            in the execution. If ``None``, it uses the current backend.
            Defaults to ``None``.

    Returns:
        ExactGeodesicTransportCG: Instantiated optimizer object.

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
        self.backend = _check_backend(backend)
        self.callback = callback
        self.n_calls_loss = 0
        self.n_calls_gradient = 0
        self.loss_fn = loss_fn

        self.v = None
        self.u = None
        self.eta = None
        self.grad = None

        if initial_parameters is not None:
            self.angles = self.backend.cast(
                initial_parameters, dtype=initial_parameters.dtype
            )
        else:
            self.angles = _generate_rbs_angles(
                self.backend.real(
                    random_statevector(
                        int(comb(nqubits, weight)), seed=seed, backend=self.backend
                    )
                ),
                "diagonal",
                backend=self.backend,
            )
            self.angles = self.backend.cast(self.angles, dtype=self.angles.dtype)

        self.x = self.angles_to_amplitudes(self.angles)
        self.circuit = hamming_weight_encoder(
            self.x,
            self.nqubits,
            self.weight,
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
                f"`loss_fn` must be either the str `exp_val` or `Callable`. Passed {type(loss_fn)}",
            )
        elif isinstance(loss_fn, str) and loss_fn != "exp_val":
            raise_error(
                ValueError,
                f"If str, `loss_fn` can only be `exp_val`. Passed {type(loss_fn)}.",
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
                    f"For backend <{self.backend}>, `hamiltonian` must be scipy sparse matrix, "
                    + f"or one of these: {self.backend.tensor_types}\n"
                    + f"passed type: {type(self.hamiltonian)}!",
                )
            if issparse(self.hamiltonian):
                self.hamiltonian = _scipy_sparse_to_backend_coo(
                    self.hamiltonian, self.backend
                )
            else:
                self.hamiltonian = self.backend.coo_matrix(self.hamiltonian)

            loss_kwargs["hamiltonian"] = self.hamiltonian

        if loss_fn == "exp_val":
            if self.hamiltonian is None:
                raise_error(
                    ValueError,
                    "For `loss_fn='exp_val'`, you must pass the hamiltonian to `loss_kwargs` "
                    + "via the dict item `{'hamiltonian': hamiltonian}`"
                )
            self.loss_fn = _loss_func_expval
            self.hamiltonian_subspace = self.get_subspace_hamiltonian()
            self.riemannian_tangent = True
            self.gradient_func = None
        else:
            backends_autodiff = ["jax", "tensorflow", "pytorch"]
            if self.backend.platform not in backends_autodiff:
                raise_error(
                    TypeError,
                    f"To use autodiff, must use one of the following backends: {backends_autodiff}",
                )
            self.hamiltonian_subspace = None
            self.gradient_func = self._gradient_func_internal

        self.loss = self._loss_internal
        self.loss_kwargs = loss_kwargs

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

        Assumes `self.hamiltonian` is in COO format of the respective backend.

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
        if platform == "jax":
            indices = self.hamiltonian.indices
            data = self.hamiltonian.data
            rows = indices[:, 0]
            cols = indices[:, 1]
        elif platform == "tensorflow":
            indices = self.hamiltonian.indices.numpy()
            data = self.hamiltonian.values.numpy()
            rows = indices[:, 0]
            cols = indices[:, 1]
        elif platform == "pytorch":
            hamilt = self.hamiltonian.coalesce()
            indices = hamilt.indices()
            values = hamilt.values()
            rows = indices[0].cpu().numpy()
            cols = indices[1].cpu().numpy()
            data = values.cpu().numpy()
        else:
            hamilt = self.hamiltonian
            rows = hamilt.row
            cols = hamilt.col
            data = hamilt.data

        tol = 1e-14
        for i_full, j_full, v in zip(rows, cols, data):
            if abs(v) <= tol:
                continue
            i = full_to_sub.get(int(i_full))
            j = full_to_sub.get(int(j_full))
            if i is None or j is None:
                continue
            if platform == "jax":
                hamilt_subspace = hamilt_subspace.at[(i, j), (j, i)].set(v)
            elif platform == "tensorflow":
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

        Sets up the internal variables `x`, `u`, `v`, and initial step size `eta`
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
                x_new,
                self.nqubits,
                self.weight,
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

                condition_b_lhs = abs((-v_new @ transported_u))
                condition_b_rhs = abs(self.c2 * (-v_prev @ u_prev))
                condition_b = condition_b_lhs <= condition_b_rhs

                if condition_a and condition_b:

                    self.x = amps_orig
                    return x_new, v_new, eta

            eta *= self.backtrack_rate

            self.angles = angles_orig
            self.x = amps_orig

        x_new = self.exponential_map_with_direction(u_prev, eta)
        self.circuit = hamming_weight_encoder(
            x_new,
            self.nqubits,
            self.weight,
            backend=self.backend,
        )
        self.angles = self.backend.cast(
            [x[0] for x in self.circuit.get_parameters()], dtype=self.backend.float64
        )
        v_new = self.tangent_vector()
        return x_new, v_new, eta

    def exponential_map_with_direction(
        self, direction: ArrayLike, eta=None
    ) -> ArrayLike:
        """Applies xponential map from current point along specified direction.

        Args:
            direction (ArrayLike): Tangent vector direction.
            eta (float, optional): Step size. Defaults to current eta.

        Returns:
            ArrayLike: Amplitudes of new point on the hypersphere.
        """
        if eta is None:
            eta = self.eta
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
            Tuple: (final_loss, losses, final_parameters)
                final_loss (float): Final loss value.
                losses (ArrayLike): Loss at each iteration.
                final_parameters (ArrayLike): Final angles.
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
                self.backend.sqrt((self.u @ self.u))
                / self.backend.sqrt((transported_u @ transported_u)),
            )
            transported_v = self.parallel_transport(self.u, -self.v, self.x)
            lt = min(
                1,
                self.backend.sqrt((self.v @ self.v))
                / self.backend.sqrt((transported_v @ transported_v)),
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
                self.x,
                self.nqubits,
                self.weight,
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
            Tuple: (final_loss, losses, final_parameters)
                final_loss (float): Final loss value.
                losses (ArrayLike): Loss at each iteration.
                final_parameters (ArrayLike): Final angles.
        """
        return self.run_egt_cg(steps=steps, tolerance=tolerance)

    def _loss_internal(self, circuit: Circuit, backend: Backend, **kwargs) -> float:
        """Wrapper function for the loss, used to update attribute `n_call_loss`
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
        Compute the gradient of `self.loss` w.r.t the trainable parameters
        stored inside self.circuit, using backpropagation of the backend specified by `platform`.
        This is used if loss != `exp_val`.

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


def _scipy_sparse_to_backend_coo(matrix, backend: Backend):
    """Convert a SciPy sparse matrix (CSR or COO) to the COO sparse
    representation supported by JAX, TensorFlow, or PyTorch.

    Args:
        matrix (scipy.sparse.csr_matrix or scipy.sparse.coo_matrix): input sparse matrix.
        backend (:class:`qibo.backends.abstract.Backend`): backend used,

    Returns:
        Backend-specific sparse tensor.
    """
    if not issparse(matrix):
        raise TypeError("Input must be a SciPy sparse matrix")

    platform = backend.platform

    if platform == "jax":
        if not isspmatrix_coo(matrix):
            matrix = matrix.tocoo()

        indices = backend.engine.stack([matrix.row, matrix.col], axis=1)
        data = matrix.data

        from jax.experimental.sparse import BCOO

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

    if platform == "pytorch":
        if not isspmatrix_coo(matrix):
            matrix = matrix.tocoo()

        row, col = matrix.row, matrix.col
        indices = backend.vstack([backend.cast(row), backend.cast(col)])
        values = matrix.data
        values = backend.cast(values)

        return backend.engine.sparse_coo_tensor(indices, values, size=matrix.shape)

    raise ValueError(
        f"Unknown platform '{platform}'. "
        "Expected one of {'jax', 'tensorflow', 'pytorch'}."
    )


def _loss_func_expval(circuit: Circuit, backend: Backend, *, hamiltonian) -> float:
    """Backend-agnostic expectation value <psi|H|psi>.

    Supports:
    - NumPy / SciPy sparse
    - JAX (BCOO)
    - TensorFlow (tf.sparse.SparseTensor)
    - PyTorch (sparse COO / CSR)

    Args:
        circuit (:class:`qibo.models.circuit.Circuit`): quantum circuit used to compute the loss.
        backend (:class:`qibo.backends.abstract.Backend`): backend for execution.
        hamiltonian: sparse hamiltonian in the backend's format.

    Assumes hamiltonian is sparse in the backend's native format
    """
    psi = backend.execute_circuit(circuit).state()
    platform = backend.platform
    if platform == "tensorflow":
        psi_col = backend.engine.reshape(psi, (-1, 1))
        h_psi = backend.engine.sparse.sparse_dense_matmul(hamiltonian, psi_col)
        h_psi = backend.engine.reshape(h_psi, (-1,))
    elif platform == "pytorch":
        h_psi = backend.engine.sparse.mm(hamiltonian, psi.unsqueeze(1)).squeeze(1)
    else:
        h_psi = hamiltonian @ psi
    return backend.real(backend.sum(backend.conj(psi) * h_psi))
