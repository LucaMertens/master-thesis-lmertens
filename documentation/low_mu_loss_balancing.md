TODO: Shorten this document

# Low-$\mu$ Loss Balancing

Low-friction terrain is rare in the dataset: cells with $\mu < 0.6$ make up only
$2.4\,\%$ of the supervised $\mu$ targets. Trained on the raw distribution the
network minimises its loss by predicting the high-$\mu$ mode everywhere and
ignoring the slippery regime that the planner most needs to detect. Two
independent levers rebalance the per-cell $\mu$ loss toward low $\mu$, and they act
on different quantities.

## Exposure: low-$\mu$ oversampling

Each training sample is a window whose loss is summed over the next
$H = 500$ horizon steps. A window is *flagged* if its **first 300 steps**
(the near horizon, $30$s) contain any cell with $\mu \le 0.6$ and
observability $o_\mu \ge 0.6$. Flagged windows are replicated deterministically
until they reach a fraction $f$ of the training index list; the replication is
identical on every rank, which keeps it compatible with the distributed sampler.

The near-horizon restriction matters. Over the full $50$ s horizon almost every
window eventually catches a stray low-$\mu$ cell, so the flag carries little
information and replication only re-shows a diffuse average. Restricting it to the
first $30\,$s lowers the natural flagged rate from $26\,\%$ to $19\,\%$ and selects
windows where the hazard is reachable soon and visible in the current view, which
is the context a planner must learn to recognise from a distance.

Oversampling changes *which cells the loss sees*, including the visual and
geometric context around a slip event, but not how hard each cell counts. Below
the natural flagged rate it is a no-op.

## Weight: label distribution smoothing

The second lever is a per-cell weight from inverse-density label distribution
smoothing (LDS). The $\mu$ targets are binned into five bins of width $0.2$, the
density is Gaussian-smoothed, and each cell receives

$$
w_\mu = \left(\frac{1}{\rho(\mu)}\right)^{\alpha},
$$

gated by the observability mask $o_\mu$ (the same score used in the RHE labels,
see [rhe_observability_tuning.md](rhe_observability_tuning.md)). The exponent
$\alpha \in [0, 1]$ tempers the inverse density: $\alpha = 1$ is full inverse
weighting, $\alpha = 0$ is uniform. It sets both the low-$\mu$ loss-mass share and
the ratio between the largest and smallest bin weight, which bounds the per-cell
gradient and thus the optimisation variance.

## Choice of hyperparameters

The two knobs are evaluated analytically (`tests/oversampling_effect.py`): for each
$(\alpha, f)$ the exact per-epoch oversampled index multiplicities are combined with
the observability-weighted per-bin cell counts to give the share of total $\mu$-loss
mass that falls on the $\mu < 0.6$ bins. The raw share with no reweighting is
$2.4\,\%$.

| $\alpha$ | $f=0.30$ | $f=0.40$ | $f=0.50$ | weight ratio |
|---|---|---|---|---|
| 0.50 | 12.7\% | 16.3\% | 20.1\% | 9$\times$ |
| **0.75** | 24.2\% | **29.9\%** | 35.6\% | 26$\times$ |
| 1.00 | 43.2\% | 50.4\% | 56.9\% | 78$\times$ |

We adopt $\alpha = 0.75$ and $f = 0.40$, giving $\approx 30\,\%$ of the $\mu$-loss
mass on low-$\mu$ terrain at a $26\times$ weight ratio.

$\alpha = 0.75$ is the balance point. $\alpha = 1.0$ assigns over half of the
gradient to a regime that is $2.4\,\%$ of the data, which degrades calibration on
the high-$\mu$ terrain the robot occupies most of the time, and its $78\times$
ratio raises gradient variance. $\alpha = 0.5$ leaves low $\mu$ at $13$ to
$26\,\%$, under-weighting the target regime. The $26\times$ ratio at $\alpha = 0.75$
is handled by `bf16-mixed` precision and gradient clipping at $10$.

$f = 0.40$ sits above the $19\,\%$ no-op floor, so it replicates rather than passes
through, and nearly doubles low-$\mu$ exposure from $2.4\,\%$ to $5.0\,\%$. Larger
fractions raise the loss share further only by replicating the same flagged windows
harder, which reduces scene diversity and invites overfitting to a few slippery
clips.

Two limits apply to this target. It is an analytical loss-mass optimum, not a
validation-error one; the per-bin validation MAE and the calibration curve are the
quantities that confirm it under training. And the share is measured over the
$300$-step flag window, so the true full-horizon share is somewhat lower, since the
trailing $200$ steps of each flagged window contribute additional high-$\mu$ cells.
