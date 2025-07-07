from qibo.models.encodings import hamming_weight_encoder
from qibo import get_backend


class ExactGeodesicTransportCG:
    """Exact Geodesic Transport with Conjugate Gradients Optimizer.

    Implements the Exact Geodesic Transport with Conjugate Gradients (EGT-CG) optimizer, a curvature-aware Riemannian optimizer designed
    specifically for variational circuits based on the Hamming-weight encoder (HWE) ansätze. It updates parameters along exact geodesic paths
    on the hyperspherical manifold defined by the HWE, combining analytic metric computation, conjugate-gradient memory, and dynamic
    learning rates for fast, globally convergent optimization.

    Args:
      n_qubits (int): Number of qubits in the quantum circuit.
      weight (int): Hamming weight to encode.
      hamiltonian (qibo.hamiltonians.Hamiltonian): Hamiltonian whose expectation
        value defines the loss to minimize.
      theta (array): Initial spherical angles parameterizing the amplitudes.
      backtrack_rate (float, optional): Backtracking rate for Wolfe condition
        line search. Defaults to 0.5 if not provided.
      geometric_gradient (bool, optional): If True, uses the geometric gradient
        estimator instead of numerical finite differences. Defaults to False.
      backend (qibo.backends.AbstractBackend, optional): Qibo backend to use for numerical
        operations. If None, the default backend is used.
      multiplicative_factor (float, optional): Scaling factor applied to the intial learning
        rate during optimization. Defaults to 1.
      c1 (float, optional): Constant for Armijo condition (sufficient decrease) in
        Wolfe line search. Defaults to 1e-3.
      c2 (float, optional): Constant for curvature condition in Wolfe line search.
        hould satisfy c1 < c2 < 1. Defaults to 0.9.

    Returns:
      ExactGeodesicTransportCG: Instantiated optimizer object.

    Adapted from: Ferreira‑Martins, A. J., Farias, R. M. S., Camilo, G., Maciel, T. O., Tosta, A., Lin, R., Alhajri, A., Haug, T., & Aolita, L. (2025, June 20).
    Variational quantum algorithms with exact geodesic transport. arXiv:2506.17395
    """

    def __init__(
        self,
        n_qubits,
        weight,
        hamiltonian,
        theta,
        backtrack_rate=None,
        geometric_gradient=False,
        backend = None,
        multiplicative_factor = 1,
        c1 = 0.0001,
        c2 = 0.9,
    ):
        self.n_qubits = n_qubits
        self.weight = weight
        self.hamiltonian = hamiltonian
        self.theta = theta
        self.backtrack_rate = backtrack_rate
        self.geometric_gradient = geometric_gradient
        self.multiplicative_factor = multiplicative_factor
        self.c1 = c1
        self.c2 = c2
        if backend is None:
            self.backend = get_backend()
        else:
            self.backend = backend

    def initialize_cg_state(self):
        """Initialize CG state."""
        self.x = self.angles_to_amplitudes(self.theta)
        self.v = self.tangent_vector()
        self.u = self.v.copy()
        # Power method learning rate
        norm_u = self.backend.np.sqrt(self.sphere_inner_product(self.u, self.u, self.x))
        loss_prev = self.loss()
        self.eta = ((1 / norm_u) * self.backend.np.arccos(
            (1 + (norm_u / (2 * loss_prev)) ** 2) ** -0.5
        )) * self.multiplicative_factor

    def angles_to_amplitudes(self, theta):
        """Convert angles to amplitudes."""
        d = len(theta) + 1
        amps = self.backend.np.zeros(d)
        for k in range(d):
            prod = 1.0
            for j in range(k):
                prod *= self.backend.np.sin(theta[j])
            if k < d - 1:
                prod *= self.backend.np.cos(theta[k])
            amps[k] = prod
        return amps

    def encoder(self):
        """Build the Hamming-weight encoder circuit for the given amplitudes."""
        amps = self.angles_to_amplitudes(self.theta)
        return hamming_weight_encoder(amps, self.n_qubits, self.weight)

    def state(self):
        """Return the statevector after encoding."""
        circuit = self.encoder()
        return circuit().state()

    def loss(self):
        """Calculte expectation of hamiltonian as loss"""
        psi = self.state()
        return self.hamiltonian.expectation(psi)

    def gradient(self):
        """Numerically compute gradient of loss wrt theta."""
        epsilon = 1e-8
        grad = self.backend.np.zeros_like(self.theta)
        for idx in range(len(self.theta)):
            theta_forward = self.theta.copy()
            theta_backward = self.theta.copy()
            theta_forward[idx] += epsilon
            theta_backward[idx] -= epsilon
            grad[idx] = (
                self.__class__(
                    self.n_qubits, self.weight, self.hamiltonian, theta_forward
                ).loss()
                - self.__class__(
                    self.n_qubits, self.weight, self.hamiltonian, theta_backward
                ).loss()
            ) / (2 * epsilon)
        return grad

    def amplitudes_to_full_state(self, amps):
        """Convert amplitudes to full state."""
        circuit = hamming_weight_encoder(amps, self.n_qubits, self.weight)
        return circuit().state()

    def geom_gradient(self):
        """Compute gradient for hamiltonian"""
        d = len(self.theta)
        grad = self.backend.np.zeros(d)
        l_psi = self.loss()
        jacobian = self.jacobian()
        g_diag = self.metric_tensor()
        psi = self.angles_to_amplitudes(self.theta)
        for j in range(d):
            varphi = g_diag[j] ** (-1 / 2) * jacobian[:, j]
            full_varphi = self.amplitudes_to_full_state(varphi)
            l_varphi = self.hamiltonian.expectation(full_varphi)
            phi = (psi + varphi) / self.backend.np.sqrt(2)
            full_phi = self.amplitudes_to_full_state(phi)
            l_phi = self.hamiltonian.expectation(full_phi)
            grad[j] = self.backend.np.sqrt(g_diag[j]) * (2 * l_phi - l_varphi - l_psi)
        return grad

    def jacobian(self):
        """Compute Jacobian of x wrt theta."""
        d = len(self.theta) + 1
        J = self.backend.np.zeros((d, d - 1))
        for i in range(d):
            for k in range(d - 1):
                if k > i:
                    J[i, k] = 0.0
                else:
                    prod = 1.0
                    for j in range(k):
                        prod *= self.backend.np.sin(self.theta[j])
                    if k == i:
                        if i < d - 1:
                            prod *= -self.backend.np.sin(self.theta[k])
                    else:
                        prod *= self.backend.np.cos(self.theta[k])
                        for j2 in range(k + 1, i):
                            prod *= self.backend.np.sin(self.theta[j2])
                        if i < d - 1:
                            prod *= self.backend.np.cos(self.theta[i])
                    J[i, k] = prod
        return J

    def metric_tensor(self):
        """Compute metric tensor."""
        theta = self.theta
        g_diag = self.backend.np.ones(len(theta))
        for j in range(1, len(theta)):
            prod = 1.0
            for l in range(j):
                prod *= self.backend.np.sin(theta[l]) ** 2
            g_diag[j] = max(prod, 1e-12)
        return g_diag

    def tangent_vector(self):
        """Compute tangent vector"""
        # Compute gradient(either finite difference or geometric)
        if self.geometric_gradient:
            grad = self.geom_gradient()
        else:
            grad = self.gradient()
        # Compute inverse metric
        g_diag = self.metric_tensor()
        inv_g = 1.0 / g_diag
        # Compute natural gradient
        nat_grad = -inv_g * grad
        jacobian = self.jacobian()
        return jacobian @ nat_grad

    def optimize_step_size(self, x_prev, u_prev, v_prev, loss_prev):
        """Adaptively determine eta using Wolfe conditions."""
        backtrack_rate = self.backtrack_rate
        if backtrack_rate is None:
            backtrack_rate = 0.5
        norm_u = self.backend.np.sqrt(self.sphere_inner_product(self.u, self.u, self.x))
        eta = self.eta
        count = 0

        while 1e-6 < eta:
            transported_u = self.parallel_transport(u_prev, u_prev, x_prev, eta)
            x_new = self.exponential_map_with_direction(u_prev, eta)
            theta_trial = self.amplitudes_to_angles(x_new)

            theta_orig = self.theta
            self.theta = theta_trial
            v_new = self.tangent_vector()
            loss_new = self.loss()
            self.theta = theta_orig

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
                return x_new, theta_trial, v_new, eta

            eta *= backtrack_rate
            count += 1

        # Fallback to last tried point
        x_new = self.exponential_map_with_direction(u_prev, eta)
        theta_trial = self.amplitudes_to_angles(x_new)
        self.theta = theta_trial
        v_new = self.tangent_vector()

        return x_new, theta_trial, v_new, eta

    def exponential_map_with_direction(self, direction, eta=None):
        """Define exponential map."""
        if eta == None:
            eta = self.eta
        norm_dir = self.backend.np.sqrt(
            self.sphere_inner_product(direction, direction, self.x)
        )
        return self.backend.np.cos(eta * norm_dir) * self.x + self.backend.np.sin(
            eta * norm_dir
        ) * (direction / norm_dir)

    def amplitudes_to_angles(self, x):
        """Convert amplitudes to angles"""
        d = len(x)
        theta = self.backend.np.zeros(d - 1)
        for i in range(d - 2):
            norm_tail = self.backend.np.linalg.norm(x[i:])
            theta[i] = (
                0.0 if norm_tail == 0 else self.backend.np.arccos(x[i] / norm_tail)
            )
        theta[-1] = self.backend.np.arctan2(x[-1], x[-2])
        return theta

    def parallel_transport(self, u, v, a, eta=None):
        """Parallel transport vector u along geodesic defined by v, starting at point a."""
        if eta == None:
            eta = self.eta
        norm_v = self.backend.np.linalg.norm(v)
        vu_dot = self.backend.np.dot(v, u)
        transported = (
            u
            - self.backend.np.sin(eta * norm_v) * (vu_dot / norm_v) * a
            + (self.backend.np.cos(eta * norm_v) - 1) * (vu_dot / (norm_v**2)) * v
        )
        return transported

    def sphere_inner_product(self, u, v, x):
        """Inner product on the tangent space at x on the sphere."""
        return self.backend.np.dot(u, v) - self.backend.np.dot(x, u) * self.backend.np.dot(x, v)

    def beta_dy(self, v_next, x_next, transported_u, st):
        """Compute Dai and Yuan Beta."""
        st_scaled_u = st * transported_u
        numerator = self.sphere_inner_product(-v_next, -v_next, x_next)
        denominator = self.sphere_inner_product(
            -v_next, st_scaled_u, x_next
        ) - self.sphere_inner_product(-self.v, self.u, self.x)
        return numerator / denominator

    def beta_hs(self, v_next, x_next, transported_u, transported_v, lt, st):
        """Compute Hestenes and Stiefel Beta."""
        numerator = self.sphere_inner_product(
            -v_next, -v_next, x_next
        ) - self.sphere_inner_product(-self.v, lt * transported_v, x_next)
        denominator = self.sphere_inner_product(
            -v_next, st * transported_u, x_next
        ) - self.sphere_inner_product(-self.v, self.u, self.x)
        return numerator / denominator

    def run_egt_cg(self, steps=10):
        """Run EGT Conjugate Gradient Optimizer."""
        self.initialize_cg_state()
        losses = []

        for i in range(steps):
            loss_prev = self.loss()
            losses.append(loss_prev)
            # Terminating Condition
            res = (
                self.sphere_inner_product(-self.v, self.u, self.x) ** 2
            ) / self.sphere_inner_product(self.u, self.u, self.x)
            if res < 1e-6:
                break

            # Save current state
            x_prev = self.x.copy()
            u_prev = self.u.copy()

            # Power method eta
            norm_u = self.backend.np.sqrt(
                self.sphere_inner_product(self.u, self.u, self.x)
            )
            self.eta = (1 / norm_u) * self.backend.np.arccos(
                (1 + (norm_u / (2 * loss_prev)) ** 2) ** -0.5
            ) * self.multiplicative_factor

            # Line search via Wolfe conditions
            x_new, theta_trial, v_new, new_eta = self.optimize_step_size(
                x_prev=x_prev, u_prev=u_prev, v_prev=self.v, loss_prev=loss_prev
            )

            # Calculate Beta
            transported_u = self.parallel_transport(self.u, self.u, self.x)
            st = min(
                1,
                self.backend.np.sqrt(self.sphere_inner_product(self.u, self.u, self.x))
                / self.backend.np.sqrt(
                    self.sphere_inner_product(transported_u, transported_u, x_new)
                ),
            )
            transported_v = self.parallel_transport(self.v, self.v, self.x)
            lt = min(
                1,
                self.backend.np.sqrt(self.sphere_inner_product(self.v, self.v, self.x))
                / self.backend.np.sqrt(
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
            self.theta = theta_trial
            self.v = v_new
            self.eta = new_eta

            # Update u
            self.u = v_new + beta_val * st * transported_u
        final_loss = self.loss()
        final_parameters = self.theta
        return final_loss, losses, final_parameters

    def __call__(self, steps=10):
        """Run the optimizer.

        Args:
            steps (int): Number of optimization steps.

        Returns:
            tuple: (final loss, loss log, final_parameters) 
        """
        return self.run_egt_cg(steps=steps)
