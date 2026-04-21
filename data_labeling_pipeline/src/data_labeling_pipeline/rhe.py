import numpy as np
from scipy.optimize import minimize


class RecedingHorizonEstimator:
    def __init__(self, dt=0.1, N=50, Px=None, Pm=None, Pw=None):
        self.dt = dt
        self.N = N
        self.Px = np.array(Px) if Px is not None else np.diag([10.0, 10.0, 10.0])
        self.Pm = np.array(Pm) if Pm is not None else np.diag([1.0, 1.0, 1.0])
        self.Pw = np.array(Pw) if Pw is not None else np.diag([1.0, 1.0, 1.0])

        self.prev_x = None
        self.prev_m = None

    def dynamics(self, state, u, m):
        """Discrete Kino-Dynamic System Model (Equation 1)."""
        px, py, theta = state
        v, omega = u
        mu, nu, _ = m

        px_next = px + mu * np.cos(theta) * v * self.dt
        py_next = py + mu * np.sin(theta) * v * self.dt
        theta_next = theta + nu * omega * self.dt

        return np.array([px_next, py_next, theta_next])

    def simulate(self, x0, U, m):
        """Simulate the horizon given initial state and inputs."""
        X = np.zeros((self.N, 3))
        X[0] = x0
        for i in range(self.N - 1):
            X[i + 1] = self.dynamics(X[i], U[i], m)
        return X

    def cost(
        self,
        opt_vars: np.ndarray,
        U: np.ndarray,
        Z: np.ndarray,
        x_guess: np.ndarray,
        m_guess: np.ndarray,
    ) -> float:
        """RHE Optimization Formulation (Equation 2).

        Args:
            opt_vars (6 x 1): Optimization variables: x_0, m.
            U (N x 2): Sequence of commanded inputs: v, omega. Passed via the args tuple.
            Z (N x 3): Sequence of raw sensor measurements: px, py, theta_true. Passed via the args tuple.
            x_guess (N x 3): Prior estimate of the initial state: px, py, theta. Passed via the args tuple.
            m_guess (N x 3): Prior estimate of the parameters: mu, nu, delta_theta. Passed via the args tuple.

        Returns:
            float: The total optimization cost.
        """
        # The predicted initial pose (px, py, theta) of the robot at the start of the current receding horizon window.
        x0 = opt_vars[0:3]
        m = opt_vars[3:6]
        d_theta = m[2]

        diff_x0 = x0 - x_guess
        diff_x0[2] = (diff_x0[2] + np.pi) % (2 * np.pi) - np.pi
        cost_x0 = diff_x0.T @ self.Px @ diff_x0

        diff_m = m - m_guess
        diff_m[2] = (diff_m[2] + np.pi) % (2 * np.pi) - np.pi
        cost_m = diff_m.T @ self.Pm @ diff_m

        X = self.simulate(x0, U, m)
        z_mapped_theta = Z[:, 2] + d_theta
        z_mapped = np.column_stack((Z[:, 0:2], z_mapped_theta))

        diff_Z = X - z_mapped
        diff_Z[:, 2] = (diff_Z[:, 2] + np.pi) % (2 * np.pi) - np.pi

        cost_measurements = np.sum([d.T @ self.Pw @ d for d in diff_Z])

        return cost_x0 + cost_m + cost_measurements

    def estimate(self, U, Z):
        """
        Run the finite horizon optimization for a single step.
        U: Commanded inputs of shape (N, 2) [v_k, omega_k]
        Z: Raw sensor measurements of shape (N, 3) [px, py, theta]
        """
        assert U.shape == (self.N, 2), f"U must have shape ({self.N}, 2)"
        assert Z.shape == (self.N, 3), f"Z must have shape ({self.N}, 3)"

        if self.prev_x is None:
            self.prev_x = Z[0].copy()
        if self.prev_m is None:
            self.prev_m = np.array([1.0, 1.0, 0.0])

        opt_vars_0 = np.concatenate((self.prev_x, self.prev_m))

        bounds = [
            (None, None),
            (None, None),
            (None, None),
            (0.0, 1.0),
            (0.0, 1.0),
            (-np.pi, np.pi),
        ]

        res = minimize(
            self.cost,
            opt_vars_0,
            args=(U, Z, self.prev_x, self.prev_m),
            method="SLSQP",
            bounds=bounds,
        )

        if not res.success:
            raise ValueError("Optimization failed:", res.message)

        opt_vars = res.x
        x0_opt = opt_vars[0:3]
        m_opt = opt_vars[3:6]

        X_opt = self.simulate(x0_opt, U, m_opt)

        self.prev_x = X_opt[1]
        self.prev_m = m_opt

        return X_opt, m_opt

    def reset(self):
        """Reset the estimator state to be able to start a new mission."""
        self.prev_x = None
        self.prev_m = None
