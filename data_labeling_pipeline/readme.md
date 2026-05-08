
## Starting points

The most important files in this module are:

- [`pipeline_beta.ipynb`](pipeline_beta.ipynb): Notebook to execute the pipeline, plot sanity-check graphs, and save a video of the resulting labels.
- [`src/data_labeling_pipeline/labeling_pipeline.py`](src/data_labeling_pipeline/labeling_pipeline.py): Defines sliding-window size and overlap, calls the RHE optimization.
- [`src/data_labeling_pipeline/rhe.py`](src/data_labeling_pipeline/rhe.py): Defines the RHE optimization problem, including the cost function and the dynamics model.
- [`src/data_labeling_pipeline/dataset_loader.py`](src/data_labeling_pipeline/dataset_loader.py): Loads the data and does some light preprocessing (inserting zero-commands for missing time steps).
- [`tests/rhe_qualitative_tests.ipynb`](tests/rhe_qualitative_tests.ipynb): Some debugging visualizations for mocked data. Can be helpful for RHE debugging.


## Problems

The forward command is almost always non-zero. The yaw rate command is very often zero. This is an issue for the RHE, because, if the command is zero, the optimizer can choose any value for the angular traversability parameter $\nu$:
```python
theta_next = theta + nu * omega_cmd * self.dt
```

### Rejected approaches

- Assume that, if the angular velocity command is zero, $\nu$ should be equal to 1 (by e.g. adding a cost term that slightly penalizes deviations of $nu$ from 1). This is probably not the best option, because it generates labels for areas where we don't have informative inputs. If the robot walks straight onto slippery ice, forcing $\nu = 1$ teaches the predictor that ice has perfect angular traversability as long as you drive straight, which is physically incorrect.
- If the angular velocity command in the current window is not higher than a certain threshold, freeze $\nu$ to its value from the previous window. If the robot walks straight for 10 seconds and transitions from asphalt to mud, $\nu$ will be frozen at the asphalt value.


### Considered approaches

- Export a confidence score alongside the labels. This could be the mean angular velocity in the window. This should be combined with a different approach, to keep the optimizer from diverging.
Then, in the neural network training, weight the loss function by this confidence: $Loss = Loss_\mu + w_\nu \cdot Loss_\nu$
- When the angular traversability is non-zero, rely on the RHE-estimated $\nu$. Else, soft-tie $\nu$ to $\mu$ by penalizing deviations of $\nu$ from $\mu$ in the cost function. This assumes that the angular traversability roughly corresponds to the linear traversability for quadrupeds. To avoid discontinuities in labels (which are hard to learn for NNs), this should probably only be done in cases where there are multiple consecutive zero-yaw commands.
- Penalize deviations of $\nu$ from its previous label inversely to the angular exitation. This is a continuous version of the hard-freeze approach.



## Unclear things

- The paper talks about a $\tilde{x}_k$ (Page 2 --> under Equation 2; or see below) as the "initial estimated state vector". *Initial* probably means the state from the end of the previous optimization window.
- On Page 2, Equation 2 (under the $\min$ term), the paper mentions the following optimization variables: $x_{k : k + N}$ (states) and $m$ (traversability parameters) .
Given $x_k$ and $m$, all following states $x_{k+1 : k + N}$ can be determined by unrolling the system model (Equation 1) forward in time, since $m$ is constant within the window. So, the optimization variables are actually $x_k$ and $m$.
We could theoretically fix $x_k = z_k$ and only optimize the parameters $m$. This would simplify the optimization problem from 6 variables to 3. However, even with ground-truth measurements, optimizing $x_k$ might be beneficial: If there are minor unmodeled kinematics or slight synchronization offsets between commands and measurements, forcing the trajectory to start exactly at $z_k$ might force the parameters $m$ to take on skewed values just to minimize the error over the rest of the horizon.


## Discrete Kino-Dynamic System Model (Equation 1)

The estimator uses a discrete kino-dynamic model running synchronously with the GNSS module at a specific sampling time $\Delta t$.

$$
x_{k+1} = \begin{bmatrix} p_{x_k} \\ p_{y_k} \\ \theta_k \end{bmatrix} + \begin{bmatrix} \mu \cdot \cos(\theta_k) & 0 \\ \mu \cdot \sin(\theta_k) & 0 \\ 0 & \nu \end{bmatrix} \begin{bmatrix} v_k \\ \omega_k \end{bmatrix} \Delta t
$$  
**State Variables (**$x_k$**):**

* $p_{x_k}$: X-position in the inertial coordinate frame at step $k$.  
* $p_{y_k}$: Y-position in the inertial coordinate frame at step $k$.  
* $\theta_k$: Heading angle in the inertial coordinate frame at step $k$.

**Control Inputs (**$u_k$**):**

* $v_k$: Commanded linear velocity.  
* $\omega_k$: Commanded angular velocity.

**Unknown Parameters (Labels to be estimated):**

* $\mu$: Linear traversability coefficient (traction).  
* $\nu$: Angular traversability coefficient.

## RHE Optimization Formulation (Equation 2)

To obtain the system states and the unknown parameters ($\mu$, $\nu$), a finite horizon optimization problem is solved over an estimation horizon $N$. The parameters are assumed to be constant within this short horizon $N$.  
**Objective Function:**

$$
\min_{x_{k:k+N}, m} ||x_k - \tilde{x}_k||_{P_x}^2 + ||m - \tilde{m}||_{P_m}^2 + \sum_{i=k}^{k+N} ||x_i - h(z_i, \Delta\theta)||_{P_w}^2
$$

**Subject to Constraints:**

1. $x_{k+1} = f(x_k, u_k)$ *(The system model from Eq. 1)*  
2. $\mu, \nu \in [0, 1]$  
3. $\Delta\theta \in [-\pi, \pi)$

**Variable Definitions for Optimization:**

* $m = [\mu, \nu, \Delta\theta]^\top$: Vector of parameters to optimize.  
* $\tilde{x}_k$: Initial estimated state vector? TODO 
* $\tilde{m}$: Estimated vector of parameters from the previous iteration.
* $N \in \mathbb{N}$: Length of the estimation horizon.  
* $z_i$: Raw sensor measurements from GNSS (position) and IMU (heading angle).  
* $\Delta\theta$: Offset between True North and compass-estimated North.  
* $h(z_i, \Delta\theta)$: Measurement equation model that maps raw sensors to the state representation.  
* $P_x, P_m, P_w$: Positive definite weighting matrices (hyperparameters for tuning trust in initial states, previous parameters, and measurements respectively).