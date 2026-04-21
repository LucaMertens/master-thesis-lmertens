import numpy as np
from data_labeling_pipeline.rhe import RecedingHorizonEstimator


def test_rhe_perfect_tracking():
    # Setup test with perfect tracking (mu=1, nu=1)
    N = 20
    dt = 0.1
    rhe = RecedingHorizonEstimator(dt=dt, N=N)

    # Generate perfect dataset
    U = np.ones((N, 2))  # Command v=1, w=1
    Z = np.zeros((N, 3))

    state = np.array([0.0, 0.0, 0.0])
    for i in range(N):
        Z[i] = state
        state = rhe.dynamics(state, U[i], [1.0, 1.0, 0.0])

    X_opt, m_opt = rhe.estimate(U, Z)

    print("Optimized labels (mu, nu, d_theta):", m_opt)

    assert np.allclose(m_opt, [1.0, 1.0, 0.0], atol=1e-2), (
        f"Failed: expected [1, 1, 0], got {m_opt}"
    )
    print("Test passed: Perfect tracking recovers mu=1, nu=1")


if __name__ == "__main__":
    test_rhe_perfect_tracking()
