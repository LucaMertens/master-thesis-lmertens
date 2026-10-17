"""Receding Horizon Estimator for the traversability parameters (mu, nu, delta_theta)."""

import numpy as np
from scipy.optimize import minimize

from .hyperparameters import (
    P_m,
    P_w,
    P_x,
    max_tie_strengh,
    obs_hi,
    obs_lo,
    v_thresh,
    w_thresh,
)


def observability(mean, lo=obs_lo, hi=obs_hi):
    """Cubic smoothstep ramp: 0 below lo, 1 above hi, smooth in between."""
    t = np.clip((mean - lo) / (hi - lo), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def integrate_route_from_twist(U, dt=0.1, x0=(0.0, 0.0, 0.0)):
    """Integrate a route from twist commands U (N, 2) with perfect parameters, from x0 (px, py, theta)."""
    rhe = RecedingHorizonEstimator(dt=dt, N=U.shape[0])
    m = (1.0, 1.0, 0.0)
    return rhe.simulate(x0, U, m)


def _normalize_angle(angle):
    """Wrap to [-pi, pi]."""
    return np.arctan2(np.sin(angle), np.cos(angle))


class RecedingHorizonEstimator:
    """Unicycle model with slip parameters: x += mu*cos(theta)*v*dt, y += mu*sin(theta)*v*dt, theta += nu*omega*dt.

    U and Z must be in a gravity-aligned, right-handed frame (z up, CCW yaw), as the loader provides.
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
        """Discrete kino-dynamic model (Eq. 1)."""
        px, py, theta = state
        v_cmd, omega_cmd = u
        # delta_theta only offsets the measured orientation
        mu, nu, delta_theta = m

        px_next = px + mu * np.cos(theta) * v_cmd * self.dt
        py_next = py + mu * np.sin(theta) * v_cmd * self.dt
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
        """RHE cost (Eq. 2). optimization_vars = [x_0 (3), m (3)], U (N, 2), Z (N, 3), priors x_prev, m_prev."""
        x0 = optimization_vars[0:3]
        m = optimization_vars[3:6]
        mu, nu, d_theta = m

        diff_x0 = x0 - x_prev  # (3,)
        diff_x0[2] = _normalize_angle(diff_x0[2])
        cost_x0 = diff_x0.T @ self.P_x @ diff_x0

        diff_m = m - m_prev  # (3,)
        diff_m[2] = _normalize_angle(diff_m[2])
        cost_m = diff_m.T @ self.P_m @ diff_m

        X = self.simulate(x0, U, m)  # (N, 3)

        Z = Z.copy()
        # true_theta = measured_theta - delta_theta
        Z[:, 2] = _normalize_angle(Z[:, 2] - d_theta)

        diff_Z = X - Z  # (N, 3)
        diff_Z[:, 2] = _normalize_angle(diff_Z[:, 2])

        cost_measurements = float(np.sum((diff_Z @ self.P_w) * diff_Z))

        v_mean = np.mean(np.abs(U[:, 0]))
        omega_mean = np.mean(np.abs(U[:, 1]))

        mu_observability = observability(v_mean)
        nu_observability = observability(omega_mean)

        # Tie strength peaks when only one of mu, nu is observable: pulls the other towards it
        coupling_strength = self.max_tie_strengh * (
            (1.0 - mu_observability) * nu_observability**2
            + (1.0 - nu_observability) * mu_observability**2
        )
        cost_tie = coupling_strength * (mu - nu) ** 2

        return cost_x0 + cost_m + cost_measurements + cost_tie

    def estimate(self, U, Z, step_size=1):
        """Optimize one window. U (N, 2) = [v, omega] commands, Z (N, 3) = [px, py, theta] measurements."""
        assert U.shape == (self.N, 2), f"U must have shape ({self.N}, 2)"
        assert Z.shape == (self.N, 3), f"Z must have shape ({self.N}, 3)"

        if self.prev_x is None:
            self.prev_x = Z[0].copy()
            # Initial heading: mean over the first window
            self.prev_x[2] = np.mean(Z[: self.N, 2])
        if self.prev_m is None:
            self.prev_m = np.array([1.0, 1.0, 0.0])

        v_mean = float(np.mean(np.abs(U[:, 0])))
        omega_mean = float(np.mean(np.abs(U[:, 1])))

        mu_observability = float(observability(v_mean))
        nu_observability = float(observability(omega_mean))

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

        return X_optimized, m_optimized, mu_observability, nu_observability, v_mean, omega_mean

    def reset(self):
        self.prev_x = None
        self.prev_m = None
