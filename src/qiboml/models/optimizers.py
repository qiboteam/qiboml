import math
from typing import Optional, Union

from numpy.typing import ArrayLike

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
      hamiltonian (:class:'qibo.hamiltonians.Hamiltonian'): Hamiltonian whose expectation
        value defines the loss to minimize.
      angles (ArrayLike): Initial hyperspherical angles parameterizing the amplitudes.
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
        A. J. Ferreira‑Martins, R. M. S. Farias, G. Camilo, T. O. Maciel, A. Tosta,
        R. Lin, A. Alhajri, T. Haug, and L. Aolita, *Quantum optimization
        with exact geodesic transport*, `arXiv:2506.17395 (2025)
        <https://arxiv.org/abs/2506.17395>`_.
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
        self.u = self.backend.cast(self.v, dtype=self.v.dtype, copy=True)
        # Power method learning rate
        norm_u = self.backend.engine.sqrt(self.backend.engine.dot(self.u, self.u))
        loss_prev = self.loss(self.circuit, self.backend, **self.loss_kwargs)
        self.eta = (1 / norm_u) * self.backend.engine.arccos(
            (1 + (norm_u / (2 * loss_prev)) ** 2) ** -0.5
        )

        # Power method learning rate
        norm_u = self.backend.sqrt(self.sphere_inner_product(self.u, self.u, self.x))
        loss_prev = self.loss()
        self.eta = (
            (1 / norm_u)
            * self.backend.arccos((1 + (norm_u / (2 * loss_prev)) ** 2) ** -0.5)
        ) * self.multiplicative_factor

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

    def encoder(self):
        """Build and return the Hamming-weight encoder circuit for the given amplitudes.

        Returns:
            circuit (qibo.models.hamming_weight_encoder): Circuit prepared with current amplitudes.
        """
        amps = self.angles_to_amplitudes(self.angles)
        self.circuit = hamming_weight_encoder(
            amps, self.nqubits, self.weight, backend=self.backend
        )

    def state(self, initial_state: ArrayLike = None, nshots=1000) -> ArrayLike:
        """Return the statevector after encoding.

        Args:
            initial_state (ArrayLike, optional): Initial statevector. Defaults to None.
            nshots (int, optional): Number of measurement shots. Defaults to 1000.

        Returns:
            ArrayLike: Statevector of the encoded quantum state.
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

        expval = self.hamiltonian.expectation(state)

        return expval

    def gradient(self, epsilon=1e-8):
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
            elif self.backend.platform == "jax":
                angles_forward.at[idx].set(angles_forward[idx] + epsilon)
                angles_backward.at[idx].set(angles_forward[idx] - epsilon)
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
            elif self.backend.platform == "jax":
                grad.at[idx].set(update)
            else:
                grad[idx] = update

            angles_forward = self.angles.copy()
            angles_backward = self.angles.copy()
            angles_forward[idx] += epsilon
            angles_backward[idx] -= epsilon
            loss_forward = self.__class__(
                self.nqubits, self.weight, self.hamiltonian, angles_forward
            ).loss()
            loss_backward = self.__class__(
                self.nqubits, self.weight, self.hamiltonian, angles_backward
            ).loss()
            grad[idx] = (loss_forward - loss_backward) / (2 * epsilon)
        return grad

    def amplitudes_to_full_state(
        self, amps: ArrayLike
    ) -> ArrayLike:  # pragma: no cover
        """Convert amplitudes to the full quantum statevector.

        Args:
            amps (ArrayLike): Amplitude vector.

        Returns:
            ArrayLike: Statevector corresponding to the given amplitudes.
        """
        circuit = hamming_weight_encoder(
            amps, self.nqubits, self.weight, backend=self.backend
        )
        return self.backend.execute_circuit(circuit).state()

    def geom_gradient(self):  # pragma: no cover
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
        grad = self.backend.zeros(d, dtype=self.backend.float64)
        l_psi = self.loss()
        jacobian = self.jacobian()

        self.jacobian = self.get_jacobian()
        g_diag = self.metric_tensor()

        l_psi = self.loss(self.circuit, self.backend, **self.loss_kwargs)
        psi_amps = self.x

        grad = self.backend.engine.zeros(d)
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
                reduced_params.at[0].set(reduced_params[0] + math.pi / 2)
            else:
                reduced_params[0] += math.pi / 2

            sins = self.backend.prod(self.backend.sin(self.angles[:j]))
        jacob = self.backend.np.zeros((dim + 1, dim), dtype=self.backend.np.float64)
        jacob = self.backend.engine.zeros((dim + 1, dim), dtype=self.backend.engine.float64)

        for j in range(dim):
            reduced_params = self.backend.engine.array(
                self.angles[j:], dtype=self.backend.engine.float64, copy=True
            )
            reduced_params[0] += self.backend.engine.pi / 2

            sins = self.backend.engine.prod(self.backend.engine.sin(self.angles[:j]))
            amps = self.angles_to_amplitudes(reduced_params)

            updates = self.backend.real(sins * amps)

            if self.backend.platform == "tensorflow":
                indices = list(range(j, jacob.shape[0]))
                indices = list(zip(indices, [j] * len(indices)))
                jacob = self.backend.engine.tensor_scatter_nd_update(
                    jacob, indices, updates
                )
            elif self.backend.platform == "jax":
                jacob.at[j:, j].set(updates)
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
            self.backend.np.prod(self.backend.np.sin(self.angles[:k]) ** 2)
            self.backend.engine.prod(self.backend.engine.sin(self.angles[:k]) ** 2)
            for k in range(len(self.angles))
        ]
        return self.backend.cast(g_diag, dtype=self.backend.float64)

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

    def optimize_step_size(
        self,
        x_prev: ArrayLike,
        u_prev: ArrayLike,
        v_prev: ArrayLike,
        loss_prev: ArrayLike,
    ) -> tuple[ArrayLike, ArrayLike, ArrayLike, float]:
        """Perform Wolfe line search to determine optimal step size eta.

        Args:
            x_prev (ArrayLike): Previous position on the sphere.
            u_prev (ArrayLike): Previous search direction.
            v_prev (ArrayLike): Previous gradient vector.
            loss_prev (float): Loss at previous position.

        Returns:
            tuple(ArrayLike, ArrayLike, ArrayLike, float): Respectively,
            updated position, angles, gradient, and step size.
        """
        backtrack_rate = self.backtrack_rate
        if backtrack_rate is None:
            backtrack_rate = 0.5
        norm_u = self.backend.sqrt(self.sphere_inner_product(self.u, self.u, self.x))
        eta = self.eta
        count = 0

        eta = self.backtrack_multiplier * self.eta

        angles_orig = self.angles
        amps_orig = self.x

        while eta > 1e-6:

            transported_u = self.parallel_transport(u_prev, u_prev, x_prev, eta)
            x_new = self.exponential_map_with_direction(u_prev, eta)

            angles_trial = self.amplitudes_to_angles(x_new)

            angles_orig = self.angles
            self.angles = self.backend.real(angles_trial)
            v_new = self.tangent_vector()
            loss_new = self.loss()
            self.angles = self.backend.real(angles_orig)
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

        if True:  # pragma: no cover
            # Fallback to last tried point
            x_new = self.exponential_map_with_direction(u_prev, eta)
            angles_trial = self.amplitudes_to_angles(x_new)
            self.angles = self.backend.real(angles_trial)
            v_new = self.tangent_vector()
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

    def exponential_map_with_direction(
        self, direction: ArrayLike, eta: Union[float, int] = None
    ) -> ArrayLike:
        """Exponential map from current point along specified direction.

        Args:
            direction (ArrayLike): Tangent vector direction.
            eta (float, optional): Step size. Defaults to current :math:`\\eta`.

        Returns:
            ArrayLike: New point on the hypersphere.
        """
        if eta is None:  # pragma: no cover
            eta = self.eta

        norm_dir = self.backend.sqrt(
            self.sphere_inner_product(direction, direction, self.x)
        )
        return self.backend.cos(eta * norm_dir) * self.x + self.backend.sin(
        norm_dir = self.backend.engine.sqrt(self.backend.engine.dot(direction, direction))
        return self.backend.engine.cos(eta * norm_dir) * self.x + self.backend.engine.sin(
            eta * norm_dir
        ) * (direction / norm_dir)

    def amplitudes_to_angles(self, x: ArrayLike) -> ArrayLike:
        """Convert amplitude vector back to hyperspherical angles.

        Args:
            x (ArrayLike): Amplitude vector.

        Returns:
            ArrayLike: Corresponding angles.
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
            elif self.backend.platform == "jax":
                angles.at[elem].set(updates)
            else:
                angles[elem] = updates

        update = self.backend.arctan2(x[-1], x[-2])
        if self.backend.platform == "tensorflow":
            angles = self.backend.engine.tensor_scatter_nd_update(
                angles, [[len(angles) - 1]], [update]
            )
        elif self.backend.platform == "jax":
            angles.at[-1].set(update)
        else:
            angles[-1] = update

        angles = self.backend.engine.zeros(d - 1)
        for i in range(d - 2):
            norm_tail = self.backend.engine.linalg.norm(x[i:])
            angles[i] = (
                0.0 if norm_tail == 0 else self.backend.engine.arccos(x[i] / norm_tail)
            )
        angles[-1] = self.backend.engine.arctan2(x[-1], x[-2])
        return angles

    def parallel_transport(
        self, u: ArrayLike, v: ArrayLike, a: ArrayLike, eta: Union[float, int] = None
    ) -> ArrayLike:
        """Parallel transport a tangent vector u along geodesic defined by v.

        Args:
            u (ArrayLike): Vector to transport.
            v (ArrayLike): Direction of geodesic.
            a (ArrayLike): Starting point on sphere.
            eta (float, optional): Step size. If ``None``, defaults to current :math:`eta`.
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

    def sphere_inner_product(
        self, u: ArrayLike, v: ArrayLike, x: ArrayLike
    ) -> ArrayLike:
        """Compute inner product on tangent space at x on the sphere.

        Args:
            u (ArrayLike): First tangent vector.
            v (ArrayLike): Second tangent vector.
            x (ArrayLike): Base point on the sphere.

        Returns:
            float: Inner product value.
        """
        return u @ v - (x @ u) * (x @ v)
        return self.backend.np.dot(u, v) - self.backend.np.dot(
            x, u
        ) * self.backend.np.dot(x, v)

    def beta_dy(
        self, v_next: ArrayLike, x_next: ArrayLike, transported_u: ArrayLike, st
    ) -> float:
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
            v_next (ArrayLike): Next gradient.
            x_next (ArrayLike): Next point.
            transported_u (ArrayLike): Parallel-transported u.
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

    def beta_hs(
        self,
        v_next: ArrayLike,
        x_next: ArrayLike,
        transported_u: ArrayLike,
        transported_v: ArrayLike,
        lt: Union[float, int],
        st: Union[float, int],
    ) -> Union[float, int]:
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
            final_parameters (ArrayLike): Final angles.
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
            x_prev = self.backend.cast(self.x, dtype=self.x.dtype, copy=True)
            u_prev = self.backend.cast(self.u, dtype=self.u.dtype, copy=True)

            # Power method eta
            norm_u = self.backend.sqrt(
                self.sphere_inner_product(self.u, self.u, self.x)
            )
            self.eta = (
                (1 / norm_u)
                * self.backend.arccos((1 + (norm_u / (2 * loss_prev)) ** 2) ** -0.5)
                * self.backend.np.arccos((1 + (norm_u / (2 * loss_prev)) ** 2) ** -0.5)
                * self.multiplicative_factor
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
                self.backend.sqrt(self.sphere_inner_product(self.u, self.u, self.x))
                / self.backend.sqrt(
                    self.sphere_inner_product(transported_u, transported_u, x_new)
                self.backend.engine.sqrt(self.backend.engine.dot(self.u, self.u))
                / self.backend.engine.sqrt(
                    self.backend.engine.dot(transported_u, transported_u)
                ),
            )
            transported_v = self.parallel_transport(self.u, -self.v, self.x)
            lt = min(
                1,
                self.backend.sqrt(self.sphere_inner_product(self.v, self.v, self.x))
                / self.backend.sqrt(
                    self.sphere_inner_product(transported_v, transported_v, x_new)
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
            self.angles = self.backend.real(angles_trial)
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

    def __call__(self, steps: int = 10, tolerance: float = 1e-8):  # pragma: no cover
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
