import numpy as np
from scipy.optimize import minimize

from .hyperparameters import (
    P_m,
    P_w,
    P_x,
    max_tie_strengh,
    v_thresh,
    w_thresh,
)


def integrate_route_from_twist(U, dt=0.1, x0=(0.0, 0.0, 0.0)):
    """Uses an instance of the RHE dynamics model to integrate a route from a sequence of twist commands.
    U: Sequence of commanded inputs: lin_vel_cmd, yaw_rate_cmd. Shape (N, 2)
    x0: Initial state: px, py, theta. Shape (3,)
    """
    rhe = RecedingHorizonEstimator(dt=dt, N=U.shape[0])
    m = (1.0, 1.0, 0.0)  # Assume perfect parameters for integration
    return rhe.simulate(x0, U, m)


def _normalize_angle(angle):
    """Normalize angle to be within [-pi, pi]."""
    return np.arctan2(np.sin(angle), np.cos(angle))


class RecedingHorizonEstimator:
    """Receding Horizon Estimator for traversability parameter estimation.

    Uses a unicycle kinematic model in ENU convention:
        px_next = px + mu * cos(theta) * v * dt
        py_next = py + mu * sin(theta) * v * dt
        theta_next = theta + nu * omega * dt

    All inputs (U, Z) must be in ENU frame (CCW-positive yaw, x=East, y=North).
    The GrandTourLoader handles the NED->ENU conversion for cpt7 odometry data.
    """

    def __init__(
        self,
        dt=0.1,
        N=8,
        P_x=P_x,
        P_m=P_m,
        P_w=P_w,
        v_thresh=v_thresh,
        w_thresh=w_thresh,
        max_tie_strengh=max_tie_strengh,
    ):
        self.dt = dt
        self.N = N
        self.P_x = np.array(P_x)
        self.P_m = np.array(P_m)
        self.P_w = np.array(P_w)

        self.v_thresh = v_thresh
        self.w_thresh = w_thresh
        self.max_tie_strengh = max_tie_strengh

        self.prev_x = None
        self.prev_m = None

    def dynamics(self, state, u, m):
        """Discrete Kino-Dynamic System Model (Equation 1)."""
        px, py, theta = state
        v_cmd, omega_cmd = u
        # delta_theta is an offset to the measured orientation, so we don't need it here.
        mu, nu, delta_theta = m

        px_next = px + mu * np.cos(theta) * v_cmd * self.dt
        py_next = py + mu * np.sin(theta) * v_cmd * self.dt
        # if np.isclose(omega, 0):
        #     theta_next = theta * nu
        # else:
        theta_next = theta + nu * omega_cmd * self.dt

        return np.array([px_next, py_next, theta_next])

    def simulate(self, x0, U, m) -> np.ndarray:
        """Simulate the horizon given initial state and inputs."""
        X = np.zeros((self.N, 3))
        X[0] = x0
        for i in range(self.N - 1):
            X[i + 1] = self.dynamics(X[i], U[i], m)
        return X

    def cost(
        self,
        optimization_vars: np.ndarray,
        U: np.ndarray,
        Z: np.ndarray,
        x_prev: np.ndarray,
        m_prev: np.ndarray,
    ) -> float:
        """RHE Optimization Formulation (Equation 2).

        Args:
            optimization_vars (6,): Optimization variables: x_0, m.
            U (N, 2): Sequence of commanded inputs: v, omega. Passed via the args tuple.
            Z (N, 3): Sequence of raw sensor measurements: px, py, theta_true. Passed via the args tuple.
            x_prev (3,): Prior estimate of the initial state: px, py, theta. Passed via the args tuple.
            m_prev (3,): Prior estimate of the parameters: mu, nu, delta_theta. Passed via the args tuple.

        Returns:
            float: The total optimization cost.
        """
        # The predicted initial pose (px, py, theta) of the robot at the start of the current receding horizon window.
        x0 = optimization_vars[0:3]
        m = optimization_vars[3:6]

        d_theta = m[2]

        diff_x0 = x0 - x_prev  # (3,)
        diff_x0[2] = _normalize_angle(diff_x0[2])
        cost_x0 = diff_x0.T @ self.P_x @ diff_x0

        diff_m = m - m_prev  # (3,)
        diff_m[2] = _normalize_angle(diff_m[2])
        cost_m = diff_m.T @ self.P_m @ diff_m

        X = self.simulate(x0, U, m)  # (N, 3)

        Z = Z.copy()
        # Measurements contain a sensor bias: true_theta = measured_theta - delta_theta
        Z[:, 2] = _normalize_angle(Z[:, 2] - d_theta)

        diff_Z = X - Z  # (N, 3)
        diff_Z[:, 2] = _normalize_angle(diff_Z[:, 2])

        cost_measurements = float(np.sum((diff_Z @ self.P_w) * diff_Z))

        v_mean = np.mean(np.abs(U[:, 0]))
        omega_mean = np.mean(np.abs(U[:, 1]))

        # Confidence in each parameter follows an exponential ramp.
        # We use a multiplier of 5.0 so that confidence reaches ~99.3% at the threshold.
        w_mu = 1.0 - np.exp(-5.0 * v_mean / self.v_thresh)
        w_nu = 1.0 - np.exp(-5.0 * omega_mean / self.w_thresh)

        # Coupling strength peaks when one parameter is observable and the other isn't.
        # Tie unobservable param towards observable one (symmetric penalty on mu-nu gap).
        coupling_strength = self.max_tie_strengh * (
            (1.0 - w_mu) * w_nu**2 + (1.0 - w_nu) * w_mu**2
        )
        cost_tie = coupling_strength * (m[0] - m[1]) ** 2

        return cost_x0 + cost_m + cost_measurements + cost_tie

    def estimate(self, U, Z, step_size=1):
        """
        Run the finite horizon optimization for a single step.
        U: Commanded inputs of shape (N, 2) [v_k, omega_k]
        Z: Raw sensor measurements of shape (N, 3) [px, py, theta]
        """
        assert U.shape == (self.N, 2), f"U must have shape ({self.N}, 2)"
        assert Z.shape == (self.N, 3), f"Z must have shape ({self.N}, 3)"

        if self.prev_x is None:
            self.prev_x = Z[0].copy()
            # The initial direction of the robot is taken from the average of the first time window.
            self.prev_x[2] = np.mean(Z[: self.N, 2])
        if self.prev_m is None:
            self.prev_m = np.array([1.0, 1.0, 0.0])

        v_mean = float(np.mean(np.abs(U[:, 0])))
        omega_mean = float(np.mean(np.abs(U[:, 1])))

        w_mu = 1.0 - np.exp(-5.0 * v_mean / self.v_thresh)
        w_nu = 1.0 - np.exp(-5.0 * omega_mean / self.w_thresh)

        bounds = [
            (None, None),
            (None, None),
            (None, None),
            (0.0, 1.0),
            (0.0, 1.0),
            (-np.pi, np.pi),
            # (0, 0),  # Ignore delta theta for now
        ]

        res = minimize(
            self.cost,
            np.concatenate((self.prev_x, self.prev_m)),
            args=(U, Z, self.prev_x, self.prev_m),
            method="SLSQP",
            bounds=bounds,
        )

        if not res.success:
            raise ValueError(f"Optimization failed: {res.message}")

        optimized_vars = res.x
        x0_optimized = optimized_vars[0:3]
        m_optimized = optimized_vars[3:6]

        X_optimized = self.simulate(x0_optimized, U, m_optimized)

        self.prev_x = X_optimized[step_size]
        self.prev_m = m_optimized

        return X_optimized, m_optimized, w_mu, w_nu, v_mean, omega_mean

    def reset(self):
        """Reset the estimator state to be able to start a new mission."""
        self.prev_x = None
        self.prev_m = None
