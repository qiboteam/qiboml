import math
from typing import Optional

from qibo.backends import _check_backend
from qibo.models.encodings import hamming_weight_encoder


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
      hamiltonian (:class:'qibo.hamiltonians.Hamiltonian'): Hamiltonian whose expectation
        value defines the loss to minimize.
      angles (ndarray): Initial hyperspherical angles parameterizing the amplitudes.
      backtrack_rate (float, optional): Backtracking rate for Wolfe condition
        line search. If ``None``, defaults to :math:`0.5`. Defaults to ``None``.
      geometric_gradient (bool, optional): If ``True``, uses the geometric gradient
          estimator instead of numerical finite differences. Defaults to ``False``.
      multiplicative_factor (float, optional): Scaling factor applied to the initial learning
          rate during optimization. Defaults to :math:`1`.
      c1 (float, optional): Constant for Armijo condition (sufficient decrease) in
          Wolfe line search. Defaults to :math:`10^{-3}`.
      c2 (float, optional): Constant for curvature condition in Wolfe line search.
          It should satisfy ``c1 < c2 < 1``. Defaults to :math:`0.9`.
      backend (:class:`qibo.backends.abstract.Backend`, optional): backend
          to be used in the execution. If ``None``, it uses the current backend.
          Defaults to ``None``.

    Returns:
      ExactGeodesicTransportCG: Instantiated optimizer object.

    References:
        A. J. Ferreira‑Martins, R. M. S. Farias, G. Camilo, T. O. Maciel, A. Tosta,
        R. Lin, A. Alhajri, T. Haug, and L. Aolita, *Variational quantum algorithms
        with exact geodesic transport*, `arXiv:2506.17395 (2025)
        <https://arxiv.org/abs/2506.17395>`_.
    """

    def __init__(
        self,
        nqubits: int,
        weight: int,
        hamiltonian,
        angles,
        backtrack_rate: Optional[float] = None,
        geometric_gradient: bool = False,
        multiplicative_factor: float = 1.0,
        c1: float = 0.0001,
        c2: float = 0.9,
        backend=None,
    ):
        self.nqubits = nqubits
        self.weight = weight
        self.hamiltonian = hamiltonian
        self.angles = angles
        self.backtrack_rate = backtrack_rate
        self.geometric_gradient = geometric_gradient
        self.multiplicative_factor = multiplicative_factor
        self.c1 = c1
        self.c2 = c2
        self.backend = _check_backend(backend)

    def initialize_cg_state(self):
        """Initialize CG state.

        Sets up the internal variables `x`, `u`, `v`, and initial step size `eta`
        based on the current angles.
        """
        self.x = self.angles_to_amplitudes(self.angles)
        self.v = self.tangent_vector()
        self.u = self.backend.cast(self.v, dtype=self.v.dtype, copy=True)

        # Power method learning rate
        norm_u = self.backend.sqrt(self.sphere_inner_product(self.u, self.u, self.x))
        loss_prev = self.loss()
        self.eta = (
            (1 / norm_u)
            * self.backend.arccos((1 + (norm_u / (2 * loss_prev)) ** 2) ** -0.5)
        ) * self.multiplicative_factor

    def angles_to_amplitudes(self, angles):
        """Convert angles to amplitudes.

        Args:
           angles (ndarray): Angles in hyperspherical coordinates.

        Returns:
            ndarray: Amplitudes calculated from the hyperspherical coordinates.
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

    def encoder(self):
        """Build and return the Hamming-weight encoder circuit for the given amplitudes.

        Returns:
            circuit (qibo.models.hamming_weight_encoder): Circuit prepared with current amplitudes.
        """
        amps = self.angles_to_amplitudes(self.angles)
        self.circuit = hamming_weight_encoder(
            amps, self.nqubits, self.weight, backend=self.backend
        )

    def state(self, initial_state=None, nshots=1000):
        """Return the statevector after encoding.

        Args:
            initial_state (ndarray, optional): Initial statevector. Defaults to None.
            nshots (int, optional): Number of measurement shots. Defaults to 1000.

        Returns:
            statevector (ndarray): Statevector of the encoded quantum state.
        """
        self.encoder()
        result = self.backend.execute_circuit(
            self.circuit, initial_state=initial_state, nshots=nshots
        )
        return result.state()

    def loss(self):
        """Loss function to be minimized.

        Given a quantum state :math:`\\ket{\\psi}` and a Hamiltonian :math:`H`, the loss
        function is defined as

        .. math::
            \\mathcal{L} = \\bra{\\psi} \\, H \\, \\ket{\\psi} \\, .

        Returns:
            float: Expectation value of ``hamiltonain``.
        """
        state = self.state()
        return self.hamiltonian.expectation_from_state(state)

    def gradient(self, epsilon=1e-8):  # pragma: no cover
        """Numerically compute gradient of loss wrt angles.

        Returns:
            ndarray: Gradient of loss w.r.t. ``angles``.
        """
        grad = self.backend.zeros_like(self.angles, dtype=self.backend.float64)
        for idx in range(len(self.angles)):
            angles_forward = self.backend.cast(
                self.angles, dtype=self.angles.dtype, copy=True
            )
            angles_backward = self.backend.cast(
                self.angles, dtype=self.angles.dtype, copy=True
            )
            if self.backend.platform == "tensorflow":
                angles_forward = self.backend.engine.tensor_scatter_nd_update(
                    angles_forward, [[idx]], [angles_forward[idx] + epsilon]
                )
                angles_backward = self.backend.engine.tensor_scatter_nd_update(
                    angles_backward, [[idx]], [angles_backward[idx] - epsilon]
                )
            else:
                angles_forward[idx] += epsilon
                angles_backward[idx] -= epsilon
            loss_forward = self.__class__(
                self.nqubits,
                self.weight,
                self.hamiltonian,
                angles_forward,
                backend=self.backend,
            ).loss()
            loss_backward = self.__class__(
                self.nqubits,
                self.weight,
                self.hamiltonian,
                angles_backward,
                backend=self.backend,
            ).loss()

            update = (loss_forward - loss_backward) / (2 * epsilon)

            if self.backend.platform == "tensorflow":
                grad = self.backend.engine.tensor_scatter_nd_update(
                    grad, [[idx]], [update]
                )
            else:
                grad[idx] = update

        return grad

    def amplitudes_to_full_state(self, amps):
        """Convert amplitudes to the full quantum statevector.

        Args:
            amps (ndarray): Amplitude vector.

        Returns:
            ndarray: Statevector corresponding to the given amplitudes.
        """
        circuit = hamming_weight_encoder(
            amps, self.nqubits, self.weight, backend=self.backend
        )
        return self.backend.execute_circuit(circuit).state()

    def geom_gradient(self):
        """Compute geometric gradient using the diagonal metric tensor and Jacobian.

        Returns:
            ndarray: Geometric gradient vector.
        """
        d = len(self.angles)
        grad = self.backend.zeros(d, dtype=self.backend.float64)
        l_psi = self.loss()
        jacobian = self.jacobian()
        g_diag = self.metric_tensor()
        psi = self.angles_to_amplitudes(self.angles)
        for j in range(d):
            varphi = g_diag[j] ** (-1 / 2) * jacobian[:, j]
            full_varphi = self.amplitudes_to_full_state(varphi)
            l_varphi = self.hamiltonian.expectation(full_varphi)
            phi = (psi + varphi) / math.sqrt(2)
            full_phi = self.amplitudes_to_full_state(phi)
            l_phi = self.hamiltonian.expectation(full_phi)
            update = self.backend.sqrt(g_diag[j]) * (2 * l_phi - l_varphi - l_psi)
            if self.backend.platform == "tensorflow":
                grad = self.backend.engine.tensor_scatter_nd_update(
                    grad, [[j]], [update]
                )
            else:
                grad[j] = update
        return grad

    def jacobian(self):
        """Compute Jacobian of amplitudes wrt angles.

        Returns:
            ndarray: Jacobian matrix.
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
            else:
                jacob[j:, j] = updates

        return jacob

    def metric_tensor(self):
        """Compute the diagonal metric tensor in hyperspherical coordinates.

        Returns:
            ndarray: Diagonal elements of the metric tensor.
        """
        g_diag = [
            self.backend.prod(self.backend.sin(self.angles[:k]) ** 2)
            for k in range(len(self.angles))
        ]
        return self.backend.cast(g_diag, dtype=self.backend.float64)

    def tangent_vector(self):
        """Compute the Riemannian gradient (tangent vector) at the current point on the hypersphere.

        Returns:
            ndarray: Tangent vector in the tangent space of the hypersphere.
        """
        # Compute gradient(either finite difference or geometric)
        grad = self.geom_gradient() if self.geometric_gradient else self.gradient()
        # Compute inverse metric
        inv_g = 1.0 / self.metric_tensor()
        # Compute natural gradient
        nat_grad = -inv_g * grad
        jacobian = self.jacobian()
        return jacobian @ nat_grad

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
        backtrack_rate = self.backtrack_rate
        if backtrack_rate is None:
            backtrack_rate = 0.5
        norm_u = self.backend.sqrt(self.sphere_inner_product(self.u, self.u, self.x))
        eta = self.eta
        count = 0

        while eta > 1e-6:
            transported_u = self.parallel_transport(u_prev, u_prev, x_prev, eta)
            x_new = self.exponential_map_with_direction(u_prev, eta)
            angles_trial = self.amplitudes_to_angles(x_new)

            angles_orig = self.angles
            self.angles = self.backend.real(angles_trial)
            v_new = self.tangent_vector()
            loss_new = self.loss()
            self.angles = self.backend.real(angles_orig)

            condition_a_lhs = loss_new - loss_prev
            condition_a_rhs = (
                self.c1 * eta * self.sphere_inner_product(-v_prev, u_prev, x_prev)
            )

            condition_a = condition_a_lhs <= condition_a_rhs

            condition_b_lhs = abs(
                self.sphere_inner_product(-v_new, transported_u, x_new)
            )
            condition_b_rhs = abs(
                self.c2 * self.sphere_inner_product(-v_prev, u_prev, x_prev)
            )

            condition_b = condition_b_lhs <= condition_b_rhs

            if condition_a and condition_b:
                # Accept step
                return x_new, angles_trial, v_new, eta

            if True:  # pragma: no cover
                eta *= backtrack_rate
                count += 1

        if True:  # pragma: no cover
            # Fallback to last tried point
            x_new = self.exponential_map_with_direction(u_prev, eta)
            angles_trial = self.amplitudes_to_angles(x_new)
            self.angles = self.backend.real(angles_trial)
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

        norm_dir = self.backend.sqrt(
            self.sphere_inner_product(direction, direction, self.x)
        )
        return self.backend.cos(eta * norm_dir) * self.x + self.backend.sin(
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
        angles = self.backend.zeros(d - 1)
        for elem in range(d - 2):
            norm_tail = self.backend.vector_norm(x[elem:])
            updates = (
                0.0 if norm_tail == 0 else self.backend.arccos(x[elem] / norm_tail)
            )
            if self.backend.platform == "tensorflow":
                angles = self.backend.engine.tensor_scatter_nd_update(
                    angles, [[elem]], [updates]
                )
            else:
                angles[elem] = updates

        update = self.backend.arctan2(x[-1], x[-2])
        if self.backend.platform == "tensorflow":
            angles = self.backend.engine.tensor_scatter_nd_update(
                angles, [[-1]], [update]
            )
        else:
            angles[-1] = update

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
        norm_v = self.backend.vector_norm(v)
        vu_dot = v @ u
        transported = (
            u
            - self.backend.sin(eta * norm_v) * (vu_dot / norm_v) * a
            + (self.backend.cos(eta * norm_v) - 1) * (vu_dot / (norm_v**2)) * v
        )
        return transported

    def sphere_inner_product(self, u, v, x):
        """Compute inner product on tangent space at x on the sphere.

        Args:
            u (ndarray): First tangent vector.
            v (ndarray): Second tangent vector.
            x (ndarray): Base point on the sphere.

        Returns:
            float: Inner product value.
        """
        return u @ v - (x @ u) * (x @ v)

    def beta_dy(self, v_next, x_next, transported_u, st):
        """Compute Dai and Yuan Beta.

        Args:
            v_next (ndarray): Next gradient.
            x_next (ndarray): Next point.
            transported_u (ndarray): Parallel-transported u.
            st (float): Scaling factor.

        Returns:
            float: Dai-Yuan beta value.
        """
        st_scaled_u = st * transported_u
        numerator = self.sphere_inner_product(-v_next, -v_next, x_next)
        denominator = self.sphere_inner_product(
            -v_next, st_scaled_u, x_next
        ) - self.sphere_inner_product(-self.v, self.u, self.x)
        return numerator / denominator

    def beta_hs(self, v_next, x_next, transported_u, transported_v, lt, st):
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
        numerator = self.sphere_inner_product(
            -v_next, -v_next, x_next
        ) - self.sphere_inner_product(-self.v, lt * transported_v, x_next)
        denominator = self.sphere_inner_product(
            -v_next, st * transported_u, x_next
        ) - self.sphere_inner_product(-self.v, self.u, self.x)
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

        for i in range(steps):
            loss_prev = self.loss()
            losses.append(loss_prev)
            # Terminating Condition
            res = (
                self.sphere_inner_product(-self.v, self.u, self.x) ** 2
            ) / self.sphere_inner_product(self.u, self.u, self.x)
            if res < tolerance:  # pragma: no cover
                break

            # Save current state
            x_prev = self.backend.cast(self.x, dtype=self.x.dtype, copy=True)
            u_prev = self.backend.cast(self.u, dtype=self.u.dtype, copy=True)

            # Power method eta
            norm_u = self.backend.sqrt(
                self.sphere_inner_product(self.u, self.u, self.x)
            )
            self.eta = (
                (1 / norm_u)
                * self.backend.arccos((1 + (norm_u / (2 * loss_prev)) ** 2) ** -0.5)
                * self.multiplicative_factor
            )

            # Line search via Wolfe conditions
            x_new, angles_trial, v_new, new_eta = self.optimize_step_size(
                x_prev=x_prev, u_prev=u_prev, v_prev=self.v, loss_prev=loss_prev
            )

            # Calculate Beta
            transported_u = self.parallel_transport(self.u, self.u, self.x)
            st = min(
                1,
                self.backend.sqrt(self.sphere_inner_product(self.u, self.u, self.x))
                / self.backend.sqrt(
                    self.sphere_inner_product(transported_u, transported_u, x_new)
                ),
            )
            transported_v = self.parallel_transport(self.v, self.v, self.x)
            lt = min(
                1,
                self.backend.sqrt(self.sphere_inner_product(self.v, self.v, self.x))
                / self.backend.sqrt(
                    self.sphere_inner_product(transported_v, transported_v, x_new)
                ),
            )
            beta_dy = self.beta_dy(
                v_next=v_new, x_next=x_new, transported_u=transported_u, st=st
            )
            beta_hs = self.beta_hs(
                v_next=v_new,
                x_next=x_new,
                transported_u=transported_u,
                transported_v=transported_v,
                lt=lt,
                st=st,
            )
            beta_val = max(0, min(beta_dy, beta_hs))

            # Accept step
            self.x = x_new
            self.angles = self.backend.real(angles_trial)
            self.v = v_new
            self.eta = new_eta

            # Update u
            self.u = v_new + beta_val * st * transported_u
        final_loss = self.loss()
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
