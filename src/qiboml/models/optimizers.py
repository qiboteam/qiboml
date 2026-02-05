from typing import Callable, Any
from qibo.models.encodings import hamming_weight_encoder, _ehrlich_algorithm
from qibo.backends import _check_backend
from qibo.quantum_info import random_statevector
from qibo.models.encodings import _generate_rbs_angles

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
        initial_parameters (ndarray): Initial hyperspherical angles parameterizing the amplitudes.
            If None, initializes from a Haar-random state.
        loss_fn (Callable): Callable to be used as the loss function.
            First two arguments (mandatory) are circuit and backend for execution.
        loss_kwargs: (dict): Additional arguments to be passed to the loss function.
            For VQE, include `hamiltonian: hamiltonian` and `type: "expval"`.
        backtrack_rate (float, optional): Backtracking rate for Wolfe condition
        line search. Defaults to :math:`0.9`.
        backtrack_multiplier (float, optional): Scaling factor applied to the initial learning
            rate for the backtrack. Usually, it's greater than 1 to guarantee a wider
            search space. Defaults to :math:`1.5`.
        c1 (float, optional): Constant for Armijo condition (sufficient decrease) in
            Wolfe line search. Defaults to :math:`10^{-3}`.
        c2 (float, optional): Constant for curvature condition in Wolfe line search.
            It should satisfy ``c1 < c2 < 1``. Defaults to :math:`0.9`.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend
            to be used in the execution. If ``None``, it uses the current backend.
            Defaults to ``None``.
        callback (Callable, optinal): callback function. First two positional arguments are
            `iter_number` and `loss_value`.
        random_seed (int, optional): random seed. Controls initialization.

    Returns:
        ExactGeodesicTransportCG: Instantiated optimizer object.

    References:
        A. J. Ferreira-Martins, R. M. S. Farias, G. Camilo, T. O. Maciel, A. Tosta, R. Lin, A. Alhajri,
        T. Haug, and L. Aolita,
        *Quantum optimization with exact geodesic transport*,
        `arXiv:2506.17395 (2025) <https://arxiv.org/abs/2506.17395>`_.
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
        c1: float = 0.0001,
        c2: float = 0.9,
        backend=None,
        callback: Callable[..., None] | None = None,
        random_seed: int | None = None,
    ):
        self.nqubits = nqubits
        self.weight = weight
        self.backtrack_rate = backtrack_rate
        self.backtrack_multiplier = backtrack_multiplier
        self.c1 = c1
        self.c2 = c2
        self.backend = _check_backend(backend)
        self.callback = callback
        self.n_calls_loss = 0
        self.n_calls_gradient = 0

        def loss_internal(circuit, backend, **kwargs):
            self.n_calls_loss += 1
            return loss_fn(circuit, backend, **kwargs)

        self.loss = loss_internal
        self.loss_kwargs = loss_kwargs

        if initial_parameters:
            self.angles = initial_parameters
        else:
            self.angles = np.array(
                _generate_rbs_angles(
                    random_statevector(
                        int(comb(nqubits, weight)), seed=random_seed
                    ).real,
                    "diagonal",
                )
            )

        self.x = self.angles_to_amplitudes(self.angles)
        self.circuit = hamming_weight_encoder(
            self.x,
            self.nqubits,
            self.weight,
            backend=self.backend,
        )

        self.riemannian_tangent = False

        if "hamiltonian" in loss_kwargs and loss_kwargs.get("type") == "expval":
            hamiltonian = loss_kwargs.get("hamiltonian")
            if not isinstance(hamiltonian, (csr_matrix, self.backend.engine.ndarray)):
                raise TypeError(
                    "`hamiltonian` must be a numpy array or scipy `csr_matrix`!"
                )
            if isinstance(hamiltonian, self.backend.engine.ndarray):
                hamiltonian = csr_matrix(hamiltonian)

            self.hamiltolian_subspace = get_subspace_hamiltonian(
                hamiltonian,
                self.nqubits,
                self.weight,
            )

            self.riemannian_tangent = True
            self.gradient_func = None

        else:
            self.hamiltolian_subspace = None

            def gradient_func_internal():
                self.n_calls_gradient += 1
                return self.geom_gradient()

            self.gradient_func = gradient_func_internal

        self.jacobian = None
        self.inverse_jacobian = None

    def initialize_cg_state(self):
        """Initialize CG state.

        Sets up the internal variables `x`, `u`, `v`, and initial step size `eta`
        based on the current angles.
        """
        self.v = self.tangent_vector()
        self.u = self.v.copy()
        # Power method learning rate
        norm_u = self.backend.engine.sqrt(self.backend.engine.dot(self.u, self.u))
        loss_prev = self.loss(self.circuit, self.backend, **self.loss_kwargs)
        self.eta = (1 / norm_u) * self.backend.engine.arccos(
            (1 + (norm_u / (2 * loss_prev)) ** 2) ** -0.5
        )

    def angles_to_amplitudes(self, angles):
        """Convert angles to amplitudes.

        Args:
           angles (ndarray): Angles in hyperspherical coordinates.

        Returns:
            ndarray: Amplitudes calculated from the hyperspherical coordinates.
        """
        d = len(angles) + 1
        amps = self.backend.engine.zeros(d)
        for k in range(d):
            prod = self.backend.engine.prod(self.backend.engine.sin(angles[:k]))
            if k < d - 1:
                prod *= self.backend.engine.cos(angles[k])
            amps[k] = prod
        return amps

    def geom_gradient(self):
        """Compute geometric gradient using the diagonal metric tensor and Jacobian.

        Returns:
            ndarray: Geometric gradient vector.
        """
        d = len(self.angles)

        self.jacobian = self.get_jacobian()
        g_diag = self.metric_tensor()

        l_psi = self.loss(self.circuit, self.backend, **self.loss_kwargs)
        psi_amps = self.x

        grad = self.backend.engine.zeros(d)
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

            phi_amps = (psi_amps + varphi_amps) / self.backend.engine.sqrt(2)
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

            grad[j] = self.backend.engine.sqrt(g_diag[j]) * (2 * l_phi - l_varphi - l_psi)

        return grad

    def get_jacobian(self):
        """Compute Jacobian of amplitudes wrt angles.

        Returns:
            ndarray: Jacobian matrix.
        """
        dim = len(self.angles)
        jacob = self.backend.engine.zeros((dim + 1, dim), dtype=self.backend.engine.float64)

        for j in range(dim):
            reduced_params = self.backend.engine.array(
                self.angles[j:], dtype=self.backend.engine.float64, copy=True
            )
            reduced_params[0] += self.backend.engine.pi / 2

            sins = self.backend.engine.prod(self.backend.engine.sin(self.angles[:j]))
            amps = self.angles_to_amplitudes(reduced_params)

            jacob[j:, j] = sins * amps

        return jacob

    def metric_tensor(self):
        """Compute the diagonal metric tensor in hyperspherical coordinates.

        Returns:
            ndarray: Diagonal elements of the metric tensor.
        """
        g_diag = [
            self.backend.engine.prod(self.backend.engine.sin(self.angles[:k]) ** 2)
            for k in range(len(self.angles))
        ]
        return self.backend.cast(g_diag, dtype="float64")

    def tangent_vector(self):
        """Compute the Riemannian gradient (tangent vector) at the current point on the hypersphere.

        Returns:
            ndarray: Tangent vector in the tangent space of the hypersphere.
        """

        if self.riemannian_tangent:

            l_psi = self.loss(self.circuit, self.backend, **self.loss_kwargs)
            psi_amps = self.x
            self.n_calls_gradient += 1

            return (2 * (l_psi * psi_amps - self.hamiltolian_subspace @ psi_amps)).real

        else:
            self.grad = self.gradient_func()

            inv_g = 1.0 / self.metric_tensor()

            nat_grad = -inv_g * self.grad

            return self.jacobian @ nat_grad

    def optimize_step_size(self, x_prev, u_prev, v_prev, loss_prev):
        """Perform Wolfe line search to determine optimal step size eta.

        Args:
            x_prev (ndarray): Previous position on the sphere.
            u_prev (ndarray): Previous search direction.
            v_prev (ndarray): Previous gradient vector.
            loss_prev (float): Loss at previous position.

        Returns:
            tuple: Respectively, updated position, angles, gradient, and step size.
        """

        eta = self.backtrack_multiplier * self.eta

        angles_orig = self.angles
        amps_orig = self.x

        while eta > 1e-6:

            transported_u = self.parallel_transport(u_prev, u_prev, x_prev, eta)
            x_new = self.exponential_map_with_direction(u_prev, eta)

            angles_trial = self.amplitudes_to_angles(x_new)

            # redefine the circuit at new amps
            self.circuit = hamming_weight_encoder(
                x_new,
                self.nqubits,
                self.weight,
                backend=self.backend,
            )
            # also, redefine the angles and amps - important for metric and jacobian
            self.angles = angles_trial
            self.x = self.angles_to_amplitudes(self.angles)

            loss_new = self.loss(self.circuit, self.backend, **self.loss_kwargs)

            condition_a_lhs = loss_new - loss_prev
            condition_a_rhs = self.c1 * eta * self.backend.engine.dot(-v_prev, u_prev)
            condition_a = condition_a_lhs <= condition_a_rhs

            if condition_a:

                v_new = self.tangent_vector()

                condition_b_lhs = abs(self.backend.engine.dot(-v_new, transported_u))
                condition_b_rhs = abs(self.c2 * self.backend.engine.dot(-v_prev, u_prev))

                condition_b = condition_b_lhs <= condition_b_rhs

                if condition_a and condition_b:
                    self.x = amps_orig
                    return x_new, angles_trial, v_new, eta

            eta *= self.backtrack_rate

            # reset original angles and amps, before looping again
            # when return happens, the angles are set to the new ones, outside
            self.angles = angles_orig
            self.x = amps_orig

        # Fallback to last tried point
        x_new = self.exponential_map_with_direction(u_prev, eta)
        angles_trial = self.amplitudes_to_angles(x_new)
        self.circuit = hamming_weight_encoder(
            x_new,
            self.nqubits,
            self.weight,
            backend=self.backend,
        )
        self.angles = angles_trial
        v_new = self.tangent_vector()
        return x_new, angles_trial, v_new, eta

    def exponential_map_with_direction(self, direction, eta=None):
        """Exponential map from current point along specified direction.

        Args:
            direction (ndarray): Tangent vector direction.
            eta (float, optional): Step size. Defaults to current eta.

        Returns:
            ndarray: New point on the hypersphere.
        """
        if eta is None:  # pragma: no cover
            eta = self.eta

        norm_dir = self.backend.engine.sqrt(self.backend.engine.dot(direction, direction))
        return self.backend.engine.cos(eta * norm_dir) * self.x + self.backend.engine.sin(
            eta * norm_dir
        ) * (direction / norm_dir)

    def amplitudes_to_angles(self, x):
        """Convert amplitude vector back to hyperspherical angles.

        Args:
            x (ndarray): Amplitude vector.

        Returns:
            ndarray: Corresponding angles.
        """
        d = len(x)
        angles = self.backend.engine.zeros(d - 1)
        for i in range(d - 2):
            norm_tail = self.backend.engine.linalg.norm(x[i:])
            angles[i] = (
                0.0 if norm_tail == 0 else self.backend.engine.arccos(x[i] / norm_tail)
            )
        angles[-1] = self.backend.engine.arctan2(x[-1], x[-2])
        return angles

    def parallel_transport(self, u, v, a, eta=None):
        """Parallel transport a tangent vector u along geodesic defined by v.

        Args:
            u (ndarray): Vector to transport.
            v (ndarray): Direction of geodesic.
            a (ndarray): Starting point on sphere.
            eta (float, optional): Step size. If ``None``, defaults to current ``eta``.
                Defaults to ``None``.

        Returns:
            ndarray: Transported vector.
        """
        if eta == None:
            eta = self.eta
        norm_v = self.backend.engine.linalg.norm(v)
        vu_dot = self.backend.engine.dot(v, u)
        transported = (
            u
            - self.backend.engine.sin(eta * norm_v) * (vu_dot / norm_v) * a
            + (self.backend.engine.cos(eta * norm_v) - 1) * (vu_dot / (norm_v**2)) * v
        )
        return transported

    def beta_dy(self, v_next, transported_u, st):
        """Compute Dai and Yuan Beta.

        Args:
            v_next (ndarray): Next gradient.
            x_next (ndarray): Next point.
            transported_u (ndarray): Parallel-transported u.
            st (float): Scaling factor.

        Returns:
            float: Dai-Yuan beta value.
        """
        numerator = self.backend.engine.dot(-v_next, -v_next)
        denominator = self.backend.engine.dot(
            -v_next,
            st * transported_u,
        ) - self.backend.engine.dot(-self.v, self.u)
        return numerator / denominator

    def beta_hs(self, v_next, transported_u, transported_v, lt, st):
        """Compute Hestenes-Stiefel conjugate gradient beta.

        Args:
            v_next (ndarray): Next gradient.
            x_next (ndarray): Next point.
            transported_u (ndarray): Parallel-transported u.
            transported_v (ndarray): Parallel-transported v.
            lt (float): Scaling factor.
            st (float): Scaling factor.

        Returns:
            float: Hestenes-Stiefel beta value.
        """
        numerator = self.backend.engine.dot(-v_next, -v_next) - self.backend.engine.dot(
            -v_next, lt * transported_v
        )
        denominator = self.backend.engine.dot(
            -v_next,
            st * transported_u,
        ) - self.backend.engine.dot(-self.v, self.u)
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
            losses (list): Loss at each iteration.
            final_parameters (ndarray): Final angles.
        """
        self.initialize_cg_state()
        losses = []
        for iter_num in range(steps):

            loss_prev = self.loss(self.circuit, self.backend, **self.loss_kwargs)
            losses.append(loss_prev)
            # Terminating Condition
            res = (self.backend.engine.dot(-self.v, self.u) ** 2) / self.backend.engine.dot(
                self.u, self.u
            )
            if res < tolerance:  # pragma: no cover
                print(f"\nOptimized converged at iteration {iter_num+1}!\n")
                break
            if self.callback is not None:
                self.callback(iter_num=iter_num + 1, loss=loss_prev, x=self.x)

            # Save current state
            x_prev = self.x.copy()
            u_prev = self.u.copy()

            # Power method eta
            norm_u = self.backend.engine.sqrt(self.backend.engine.dot(self.u, self.u))
            self.eta = (1 / norm_u) * self.backend.engine.arccos(
                (1 + (norm_u / (2 * loss_prev)) ** 2) ** -0.5
            )

            # Line search via Wolfe conditions and step!
            x_new, angles_trial, v_new, new_eta = self.optimize_step_size(
                x_prev=x_prev, u_prev=u_prev, v_prev=self.v, loss_prev=loss_prev
            )
            transported_u = self.parallel_transport(self.u, self.u, self.x)

            st = min(
                1,
                self.backend.engine.sqrt(self.backend.engine.dot(self.u, self.u))
                / self.backend.engine.sqrt(
                    self.backend.engine.dot(transported_u, transported_u)
                ),
            )
            transported_v = self.parallel_transport(self.u, -self.v, self.x)
            lt = min(
                1,
                self.backend.engine.sqrt(self.backend.engine.dot(self.v, self.v))
                / self.backend.engine.sqrt(
                    self.backend.engine.dot(transported_v, transported_v)
                ),
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
            self.angles = angles_trial
            self.v = v_new
            self.eta = new_eta
            self.u = v_new + beta_val * st * transported_u

            self.circuit = hamming_weight_encoder(
                self.x,
                self.nqubits,
                self.weight,
                backend=self.backend,
            )

        final_loss = self.loss(self.circuit, self.backend, **self.loss_kwargs)
        losses.append(final_loss)
        final_parameters = self.angles
        return final_loss, losses, final_parameters

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


def get_subspace_hamiltonian(
    hamiltonian_sparse,
    nqubits,
    weight,
):
    """
    Returns the hamiltonian sliced on the $\comb{n}{k}$-dimensional subspace.
    """

    subspace_dim = int(comb(nqubits, weight))

    initial_string = np.array([1] * weight + [0] * (nqubits - weight))
    lexicographical_order = _ehrlich_algorithm(initial_string, False)
    lexicographical_order.sort()

    coo_matrix = hamiltonian_sparse.tocoo()

    # dict with non-zero values of hamiltonian
    # each non-zero element in position (i, j) is identified with the following format:
    # {
    #     i: {
    #         j: element
    #     }
    # }
    sparse_hamilt_nonzero_elements = {}

    for i, j, v in zip(coo_matrix.row, coo_matrix.col, coo_matrix.data):
        if abs(v) > 1e-14:
            if i not in sparse_hamilt_nonzero_elements:
                sparse_hamilt_nonzero_elements[i] = {j: v}
            else:
                sparse_hamilt_nonzero_elements[i][j] = v

    # ================================================

    basis_states_subspace = [int(bitstring, 2) for bitstring in lexicographical_order]

    hamiltonian_subspace = np.zeros((subspace_dim, subspace_dim), dtype=np.complex128)

    for i in range(subspace_dim):

        i_in_full_matrix = basis_states_subspace[i]

        if i_in_full_matrix in sparse_hamilt_nonzero_elements:

            for j in range(i, subspace_dim):

                j_in_full_matrix = basis_states_subspace[j]

                if j_in_full_matrix in sparse_hamilt_nonzero_elements.get(
                    i_in_full_matrix
                ):

                    hamiltonian_subspace[i][j] = sparse_hamilt_nonzero_elements.get(
                        i_in_full_matrix
                    ).get(j_in_full_matrix)
                    hamiltonian_subspace[j][i] = hamiltonian_subspace[i][j]

    return hamiltonian_subspace


def get_amps_values_uniform_at_excitations(nqubits, weight, max_val):

    dim = int(comb(nqubits, weight))

    # remaining norm of the amplitudes except the hartree one
    remaining_norm = 1 - max_val**2

    n = dim - 1
    remaining_amps = [np.sqrt(remaining_norm / n) for _ in range(n)]

    amps = np.array([max_val] + remaining_amps)

    values_inverted_endiannes = np.zeros_like(amps)
    initial_string = np.array([1] * weight + [0] * (nqubits - weight))
    basis_states_ordered = _ehrlich_algorithm(initial_string, False)
    for i, basis in enumerate(basis_states_ordered):
        index_to_bring = basis_states_ordered.index(basis[::-1])
        values_inverted_endiannes[i] = amps[index_to_bring]

    amps = values_inverted_endiannes

    return amps
