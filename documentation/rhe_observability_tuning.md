# RHE Observability, Tying, and Hyperparameter Tuning

The Receding Horizon Estimator (RHE) fits one parameter set
$m = (\mu, \nu, \delta\theta)$ per sliding window from the commanded twist $U$ and
the odometry $Z$. $\mu$ scales commanded linear velocity, $\nu$ scales commanded
yaw rate. A parameter is identifiable only when its command is non-zero: if
$v_k \approx 0$ over the window, $\mu$ multiplies a near-zero input and any value
fits the data equally well, so the estimate is unconstrained. The same holds for
$\nu$ under low yaw rate.

## Observability

We try to quantify this with an observability score, a cubic smoothstep
on the mean absolute command magnitude. With $\bar v = \frac{1}{N}\sum_k |v_k|$ and
$\bar\omega = \frac{1}{N}\sum_k |\omega_k|$,

$$
g(x) = t^2(3 - 2t), \qquad t = \mathrm{clip}\!\left(\frac{x - x_\text{lo}}{x_\text{hi} - x_\text{lo}}, 0, 1\right),
$$

$$
o_\mu = g(\bar v), \qquad o_\nu = g(\bar\omega),
$$

with $x_\text{lo} = 0.01$, $x_\text{hi} = 0.1$ (m/s for $v$, rad/s for $\omega$).
The score is $0$ below $x_\text{lo}$, $1$ above $x_\text{hi}$, and rises smoothly
in between. A previous version used an exponential ramp instead; the smoothstep
gives a sharper cutoff at the ends without the exponential's long tail of
intermediate scores.

The score is used twice: it sets the tying strength inside the RHE (below), and
$o_\mu$ is saved per label and reused as a soft mask on the $\mu$ regression loss of
the traversability network, suppressing gradient from unobservable windows.

## Tying Strength

When one parameter is observable and the other is not, the unobserved one is pulled
toward the observed one, encoding the prior that linear and angular slip correlate
on the same terrain. The window cost adds

$$
J_\text{tie} = \lambda_\text{tie}\,
\Big[(1 - o_\mu)\,o_\nu^2 + (1 - o_\nu)\,o_\mu^2\Big]\,(\mu - \nu)^2 .
$$

The bracket peaks when one score is high and the other low; when both are high or
both low it vanishes and the estimates follow the measurement residual. The penalty
acts on $(\mu - \nu)^2$, shrinking the gap without fixing either value.
$\lambda_\text{tie}$ (`max_tie_strengh`) sets its strength relative to the other
cost weights.

## Synthetic Benchmark

`tests/rhe_benchmark.py` runs the RHE over synthetic scenarios with known
ground-truth parameters (constant terrain, transitions, straight-only and turn-only
motion, full stops, mixed excitation, a realistic mission, and a noise sweep) and
reports MAE and RMSE of $\mu$ and $\nu$ against the per-window truth, plus the mean
observability scores.

## Hyperparameter Optimization

The cost weights are calibrated by differential evolution
(`tests/optimize_hyperparameters.py`), minimizing estimation error on the realistic
mission scenario. Eight parameters are optimized with reduced degrees of freedom
from the planar $x$/$y$ symmetry: the tied $P_x[0{=}1]$ and $P_x[2]$, the full $P_m$
diagonal, the tied $P_w[0{=}1]$ and $P_w[2]$, and $\lambda_\text{tie}$. Bounds span
$[10^{-3}, 5]$; the ramp bounds $x_\text{lo}, x_\text{hi}$ are held fixed.

The objective is an observability-weighted parameter error,

$$
\frac{1}{|W|}\sum_{w} \tfrac12\big(\max(0.1, o_\mu)\,|\hat\mu - \mu^\star|
+ \max(0.1, o_\nu)\,|\hat\nu - \nu^\star|\big),
$$

optionally blended with state-trajectory RMSE via `param_weight`. Weighting by
observability lets well-excited windows dominate the fit; the $0.1$ floor keeps the
rest contributing. A log-space prior pulls the solution toward the current baseline.

The current values are $P_x = \mathrm{diag}(1.5, 1.5, 1.0)$,
$P_m = \mathrm{diag}(0.2, 0.2, 1.5)$, $P_w = \mathrm{diag}(1.2, 1.2, 1.0)$,
$\lambda_\text{tie} = 1.3$, with $N = 10$, $\mathrm{d}t = 0.1$ s, and step size $2$
(see [labeling_hyperparameters.tex](labeling_hyperparameters.tex)).
