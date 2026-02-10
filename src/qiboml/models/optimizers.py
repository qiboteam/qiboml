import math
from typing import Optional, Union, Callable, Any

from numpy.typing import ArrayLike

from qibo.models.encodings import hamming_weight_encoder, _ehrlich_algorithm
from qibo.backends import _check_backend, Backend
from qibo.quantum_info import random_statevector
from qibo.models.encodings import _generate_rbs_angles
from qibo.config import raise_error

import numpy as np

from scipy.sparse import csr_matrix
from scipy.special import comb


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
        initial_parameters (ArrayLike): Initial hyperspherical angles parameterizing the amplitudes.
            If None, initializes from a Haar-random state.
        loss_fn (str or Callable): if str, only possibility is `exp_val`, for expectation value loss.
            It can also be a Callable to be used as the loss function.
            First two arguments (mandatory) are circuit and backend for execution.
        loss_kwargs: (dict): Additional arguments to be passed to the loss function.
            For VQE, include `hamiltonian: hamiltonian` and `type: "exp_val"`.
        backtrack_rate (float, optional): Backtracking rate for Wolfe condition
        line search. Defaults to :math:`0.9`.
        backtrack_multiplier (float, optional): Scaling factor applied to the initial learning
            rate for the backtrack. Usually, it's greater than 1 to guarantee a wider
            search space. Defaults to :math:`1.5`.
        c1 (float, optional): Constant for Armijo condition (sufficient decrease) in
            Wolfe line search. Defaults to :math:`10^{-3}`.
        c2 (float, optional): Constant for curvature condition in Wolfe line search.
            It should satisfy ``c1 < c2 < 1``. Defaults to :math:`0.9`.
        callback (Callable, optinal): callback function. First two positional arguments are
            `iter_number` and `loss_value`.
        seed (int, optional): random seed. Controls initialization.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend
            to be used in the execution. If ``None``, it uses the current backend.
            Defaults to ``None``.

    Returns:
        ExactGeodesicTransportCG: Instantiated optimizer object.

    References:
        A. J. Ferreira‑Martins, R. M. S. Farias, G. Camilo, T. O. Maciel, A. Tosta,
        R. Lin, A. Alhajri, T. Haug, and L. Aolita, *Quantum optimization
        with exact geodesic transport*, `arXiv:2506.17395 (2025)
        <https://arxiv.org/abs/2506.17395>`_.
    """

    def __init__(
        self,
        nqubits: int,
        weight: int,
        initial_parameters,
        loss_fn: Callable[..., tuple[float, Any]],
        loss_kwargs: dict | None = None,
        backtrack_rate: float = 0.9,
        backtrack_multiplier: float = 1.5,
        backtrack_min_lr: float = 1e-6,
        c1: float = 0.0001,
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
            )
            self.angles = self.backend.cast(self.angles, dtype=self.angles.dtype)

        self.x = self.angles_to_amplitudes(self.angles)
        self.circuit = hamming_weight_encoder(
            self.x,
            self.nqubits,
            self.weight,
            backend=self.backend,
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

        if "hamiltonian" in loss_kwargs and loss_fn == "exp_val":

            def loss_func(circuit, backend, *, hamiltonian):
                state = backend.execute_circuit(circuit).state()
                # print()
                # circuit.draw()
                # print()
                # print("parameters", circuit.get_parameters())
                # print("\tstate:")
                # print(backend.real(state))
                return backend.real(backend.conj(state) @ hamiltonian @ state)

            loss_fn = loss_func

            hamiltonian = loss_kwargs.get("hamiltonian")
            if not isinstance(hamiltonian, (csr_matrix, *self.backend.tensor_types)):
                raise_error(
                    TypeError, "`hamiltonian` must be ArrayLike or scipy `csr_matrix`!"
                )
            if not isinstance(hamiltonian, csr_matrix):
                hamiltonian = csr_matrix(hamiltonian)

            self.hamiltonian_subspace = self.get_subspace_hamiltonian(
                hamiltonian,
                self.nqubits,
                self.weight,
            )

            self.riemannian_tangent = True
            self.gradient_func = None

            if self.backend.platform == "pytorch":
                coo = hamiltonian.tocoo()
                indices = self.backend.engine.from_numpy(
                    np.vstack((coo.row, coo.col))
                ).long()
                values = self.backend.engine.from_numpy(coo.data)
                shape = self.backend.engine.Size(coo.shape)
                loss_kwargs["hamiltonian"] = self.backend.engine.sparse_coo_tensor(
                    indices, values, shape
                )

        else:
            self.hamiltonian_subspace = None

            def gradient_func_internal():
                self.n_calls_gradient += 1
                return self.geom_gradient()

            self.gradient_func = gradient_func_internal

        def loss_internal(circuit, backend, **kwargs):
            self.n_calls_loss += 1
            return loss_fn(circuit, backend, **kwargs)

        self.loss = loss_internal
        self.loss_kwargs = loss_kwargs

        self.jacobian = None
        self.inverse_jacobian = None

    def get_subspace_hamiltonian(
        self,
        hamiltonian_sparse,
        nqubits,
        weight,
    ):
        """
        Returns the hamiltonian sliced on the (n_choose_k)-dimensional subspace.
        """

        subspace_dim = int(comb(nqubits, weight))

        # initial_string = self.backend.cast([1] * weight + [0] * (nqubits - weight), dtype=self.backend.int64)
        initial_string = [1] * weight + [0] * (nqubits - weight)

        lexicographical_order = _ehrlich_algorithm(initial_string, False)
        lexicographical_order.sort()

        coo_matrix = hamiltonian_sparse.tocoo()

        sparse_hamilt_nonzero_elements = {}

        for i, j, v in zip(coo_matrix.row, coo_matrix.col, coo_matrix.data):
            if abs(v) > 1e-14:
                if i not in sparse_hamilt_nonzero_elements:
                    sparse_hamilt_nonzero_elements[i] = {j: v}
                else:
                    sparse_hamilt_nonzero_elements[i][j] = v

        # ================================================

        basis_states_subspace = [
            int(bitstring, 2) for bitstring in lexicographical_order
        ]

        # hamiltonian_subspace = self.backend.zeros(
        #     (subspace_dim, subspace_dim), dtype=self.x.dtype
        # )

        # for i in range(subspace_dim):

        #     i_in_full_matrix = basis_states_subspace[i]

        #     if i_in_full_matrix in sparse_hamilt_nonzero_elements:

        #         for j in range(i, subspace_dim):

        #             j_in_full_matrix = basis_states_subspace[j]

        #             if j_in_full_matrix in sparse_hamilt_nonzero_elements.get(
        #                 i_in_full_matrix
        #             ):
        #                 value = sparse_hamilt_nonzero_elements.get(
        #                     i_in_full_matrix
        #                 ).get(j_in_full_matrix)

        #                 if self.backend.platform == "jax":
        #                     hamiltonian_subspace = hamiltonian_subspace.at[
        #                         (i, j), (j, i)
        #                     ].set(value)

        #                 else:
        #                     hamiltonian_subspace[i][j] = value
        #                     hamiltonian_subspace[j][i] = hamiltonian_subspace[i][j]

        # hamiltonian_subspace = self.backend.zeros(
        #     (subspace_dim, subspace_dim), dtype=self.x.dtype
        # )

        # if self.backend.platform == "jax":
        #     for i in range(subspace_dim):
        #         i_full = basis_states_subspace[i]
        #         row = sparse_hamilt_nonzero_elements.get(i_full)
        #         if row is None:
        #             continue
        #         for j in range(i, subspace_dim):
        #             j_full = basis_states_subspace[j]
        #             value = row.get(j_full)
        #             if value is None:
        #                 continue
        #             hamiltonian_subspace = hamiltonian_subspace.at[(i, j), (j, i)].set(
        #                 value
        #             )
        # else:
        #     for i in range(subspace_dim):
        #         i_full = basis_states_subspace[i]
        #         row = sparse_hamilt_nonzero_elements.get(i_full)
        #         if row is None:
        #             continue

        #         for j in range(i, subspace_dim):
        #             j_full = basis_states_subspace[j]
        #             value = row.get(j_full)
        #             if value is None:
        #                 continue

        #             hamiltonian_subspace[i, j] = value
        #             hamiltonian_subspace[j, i] = value

        hamiltonian_subspace = self.backend.zeros(
            (subspace_dim, subspace_dim), dtype=self.x.dtype
        )

        if self.backend.platform == "jax":
            for i in range(subspace_dim):
                i_full = basis_states_subspace[i]
                row = sparse_hamilt_nonzero_elements.get(i_full)
                if row is None:
                    continue

                for j in range(i, subspace_dim):
                    j_full = basis_states_subspace[j]
                    value = row.get(j_full)
                    if value is None:
                        continue

                    hamiltonian_subspace = hamiltonian_subspace.at[(i, j), (j, i)].set(
                        value
                    )

        elif self.backend.platform == "tensorflow":
            for i in range(subspace_dim):
                i_full = basis_states_subspace[i]
                row = sparse_hamilt_nonzero_elements.get(i_full)
                if row is None:
                    continue

                for j in range(i, subspace_dim):
                    j_full = basis_states_subspace[j]
                    value = row.get(j_full)
                    if value is None:
                        continue
                    value = self.backend.cast(value, dtype=self.x.dtype)
                    indices = self.backend.engine.constant(
                        [[i, j], [j, i]], dtype=self.backend.int32
                    )
                    updates = self.backend.engine.stack([value, value])

                    hamiltonian_subspace = self.backend.engine.tensor_scatter_nd_update(
                        hamiltonian_subspace, indices, updates
                    )

        else:
            for i in range(subspace_dim):
                i_full = basis_states_subspace[i]
                row = sparse_hamilt_nonzero_elements.get(i_full)
                if row is None:
                    continue

                for j in range(i, subspace_dim):
                    j_full = basis_states_subspace[j]
                    value = row.get(j_full)
                    if value is None:
                        continue

                    hamiltonian_subspace[i, j] = value
                    hamiltonian_subspace[j, i] = value

        return hamiltonian_subspace

    def initialize_cg_state(self):
        """Initialize CG state.

        Sets up the internal variables `x`, `u`, `v`, and initial step size `eta`
        based on the current angles.
        """
        self.v = self.tangent_vector()
        self.u = self.backend.cast(self.v, dtype=self.v.dtype, copy=True)
        # Power method learning rate
        norm_u = self.backend.sqrt((self.u @ self.u))
        loss_prev = self.loss(self.circuit, self.backend, **self.loss_kwargs)
        self.eta = (1 / norm_u) * self.backend.arccos(
            (1 + (norm_u / (2 * loss_prev)) ** 2) ** -0.5
        )

    def angles_to_amplitudes(self, angles):
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

    def geom_gradient(self):
        """Compute geometric gradient using the diagonal metric tensor and Jacobian.

        Returns:
            ArrayLike: Geometric gradient vector.
        """
        d = len(self.angles)

        self.jacobian = self.get_jacobian()
        g_diag = self.metric_tensor()

        l_psi = self.loss(self.circuit, self.backend, **self.loss_kwargs)
        psi_amps = self.x

        grad = self.backend.zeros(d)
        for j in range(d):

            varphi_amps = g_diag[j] ** (-1 / 2) * self.jacobian[:, j]
            l_varphi = self.loss(
                hamming_weight_encoder(
                    varphi_amps,
                    self.nqubits,
                    self.weight,
                    backend=self.backend,
                ),
                self.backend,
                **self.loss_kwargs,
            )

            phi_amps = (psi_amps + varphi_amps) / math.sqrt(2)
            l_phi = self.loss(
                hamming_weight_encoder(
                    phi_amps,
                    self.nqubits,
                    self.weight,
                    backend=self.backend,
                ),
                self.backend,
                **self.loss_kwargs,
            )

            grad[j] = self.backend.sqrt(g_diag[j]) * (2 * l_phi - l_varphi - l_psi)

        return grad

    def get_jacobian(self):
        """Compute Jacobian of amplitudes wrt angles.

        Returns:
            ArrayLike: Jacobian matrix.
        """
        # dim = len(self.angles)
        # jacob = self.backend.zeros((dim + 1, dim), dtype=self.backend.float64)

        # for j in range(dim):
        #     reduced_params = self.backend.cast(
        #         self.angles[j:], dtype=self.backend.float64, copy=True
        #     )
        #     reduced_params[0] += math.pi / 2

        #     sins = self.backend.prod(self.backend.sin(self.angles[:j]))
        #     amps = self.angles_to_amplitudes(reduced_params)

        #     jacob[j:, j] = sins * amps

        # return jacob

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
                reduced_params = reduced_params.at[0].set(reduced_params[0] + math.pi / 2)
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

    def metric_tensor(self):
        """Compute the diagonal metric tensor in hyperspherical coordinates.

        Returns:
            ArrayLike: Diagonal elements of the metric tensor.
        """
        g_diag = [
            self.backend.prod(self.backend.sin(self.angles[:k]) ** 2)
            for k in range(len(self.angles))
        ]
        return self.backend.cast(g_diag, dtype="float64")

    def tangent_vector(self):
        """Compute the Riemannian gradient (tangent vector) at the current point on the hypersphere.

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
        return self.jacobian @ nat_grad

    def regularization(self, angles):

        # print(angles)

        # angles = angles % (2 * math.pi)
        condition = self.backend.abs(self.backend.sin(angles[:-1])) < 1e-3
        # angles[:-1] = self.backend.where(condition, math.pi / 2, angles[:-1])
        # return self.angles_to_amplitudes(angles)

        # print(angles)
        # print(self.backend.abs(self.backend.sin(angles[:-1])))
        # print(condition)

        updated = self.backend.where(condition, math.pi / 2, angles[:-1])
        return self.angles_to_amplitudes(
            self.backend.concatenate([updated, angles[-1:]], axis=0)
        )

    def optimize_step_size(self, x_prev, u_prev, v_prev, loss_prev):
        """Perform Wolfe line search to determine optimal step size eta.

        Args:
            x_prev (ArrayLike): Previous position on the sphere.
            u_prev (ArrayLike): Previous search direction.
            v_prev (ArrayLike): Previous gradient vector.
            loss_prev (float): Loss at previous position.

        Returns:
            tuple: Respectively, updated position, angles, gradient, and step size.
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

            # redefine the circuit at new amps
            self.circuit = hamming_weight_encoder(
                x_new,
                self.nqubits,
                self.weight,
                backend=self.backend,
            )
            # also, redefine the angles and amps - important for metric and jacobian
            self.angles = self.backend.cast(
                [x[0] for x in self.circuit.get_parameters()],
                dtype=self.backend.float64,
            )
            self.x = x_new

            # print(x_prev)
            # print(u_prev)
            # print(x_new)
            # print()

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

            # reset original angles and amps, before looping again
            # when return happens, the angles are set to the new ones, outside
            self.angles = angles_orig
            self.x = amps_orig

        # Fallback to last tried point
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

    def exponential_map_with_direction(self, direction, eta=None):
        """Exponential map from current point along specified direction.

        Args:
            direction (ArrayLike): Tangent vector direction.
            eta (float, optional): Step size. Defaults to current eta.

        Returns:
            ArrayLike: New point on the hypersphere.
        """
        if eta is None:  # pragma: no cover
            eta = self.eta

        norm_dir = self.backend.sqrt((direction @ direction))
        x_new =  self.backend.cos(eta * norm_dir) * self.x + self.backend.sin(
            eta * norm_dir
        ) * (direction / norm_dir)
        
        return x_new

    def amplitudes_to_angles(self, x):
        """Convert amplitude vector back to hyperspherical angles.

        Args:
            x (ArrayLike): Amplitude vector.

        Returns:
            ArrayLike: Corresponding angles.
        """

        # d = len(x)
        # angles = self.backend.zeros(d - 1)
        # for i in range(d - 2):
        #     norm_tail = self.backend.vector_norm(x[i:])
        #     angles[i] = 0.0 if norm_tail == 0 else self.backend.arccos(x[i] / norm_tail)

        # angles[-1] = self.backend.arctan2(x[-1], x[-2])
        # return angles

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

        # print("saida angles:", angles)

        return angles

    def parallel_transport(self, u, v, a, eta=None):
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
        if eta == None:
            eta = self.eta
        norm_v = self.backend.vector_norm(v)
        vu_dot = v @ u
        transported = (
            u
            - self.backend.sin(eta * norm_v) * (vu_dot / norm_v) * a
            + (self.backend.cos(eta * norm_v) - 1) * (vu_dot / (norm_v**2)) * v
        )
        return transported

    def beta_dy(self, v_next, transported_u, st):
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

    def beta_hs(self, v_next, transported_u, transported_v, lt, st):
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

    def run_egt_cg(self, steps: int = 10, tolerance: float = 1e-8):
        """Run the EGT-CG optimizer for a specified number of steps.

        Args:
            steps (int, optional): Number of optimization iterations. Defaults to :math:`10`.
            tolerance (float, optional): Maximum tolerance for the residue of the gradient update.
                Defaults to :math:`10^{-8}`.

        Returns:
            tuple: (final_loss, losses, final_parameters)
                final_loss (float): Final loss value.
                losses (ArrayLike): Loss at each iteration.
                final_parameters (ArrayLike): Final angles.
        """
        self.initialize_cg_state()
        losses = []
        for iter_num in range(steps):

            # print(f"iter {iter_num}")
            # print(self.x)
            # print()

            loss_prev = self.loss(self.circuit, self.backend, **self.loss_kwargs)
            losses.append(loss_prev)

            # print(f"angles: {self.circuit.get_parameters()}")
            # print(loss_prev)

            res = ((-self.v @ self.u) ** 2) / (
                self.u @ self.u
            )
            if res < tolerance:  # pragma: no cover
                print(f"\nOptimized converged at iteration {iter_num+1}!\n")
                break

            if self.callback is not None:
                self.callback(iter_num=iter_num + 1, loss=loss_prev, x=self.x)

            x_prev = self.backend.cast(self.x, dtype=self.x.dtype, copy=True)
            u_prev = self.backend.cast(self.u, dtype=self.u.dtype, copy=True)

            norm_u = self.backend.sqrt((self.u @ self.u))
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

    def __call__(self, steps: int = 10, tolerance: float = 1e-8):
        """Run the optimizer.

        Args:
            steps (int): Number of optimization steps.
            tolerance (float): Maximum tolerance for the residue of the gradient update.
                Defaults to :math:`10^{-8}.`

        Returns:
            tuple: Respectively, final loss, loss log, and final parameters.
        """
        return self.run_egt_cg(steps=steps, tolerance=tolerance)
