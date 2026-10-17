# data_labeling_pipeline

Estimates per-timestep linear and angular traversability (mu, nu) from the extracted odometry and commands with a Receding Horizon Estimator (RHE), following WayFASTER. These are the training labels for the network.

- `src/data_labeling_pipeline/labeling_pipeline.py`: window size and overlap, calls the RHE.
- `src/data_labeling_pipeline/rhe.py`: cost function and dynamics model.
- `src/data_labeling_pipeline/dataset_loader.py`: loads data, inserts zero commands for missing steps.
- The previous window's end state is used as the prior $\tilde{x}_k$ (undefined in the paper). The window start state $x_k$ is optimized, not fixed to the measurement, so command/measurement sync offsets do not skew mu.
- `rhe_states` is the RHE state estimate, not ground truth odometry.
- Known problem: with zero yaw command nu is unobservable, since `theta + nu * omega * dt` does not depend on nu. Pulling nu to 1 or freezing it to the previous window were tried and rejected (wrong labels on ice, or stale after a surface change).

## Usage

```bash
uv run python preprocessing/pipeline.py --mission ETH-1
```
