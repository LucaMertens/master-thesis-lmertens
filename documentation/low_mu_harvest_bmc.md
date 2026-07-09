# Low-$\mu$ View Harvesting and Balanced MSE

Reweighting the $\mu$ loss (oversampling, LDS; see
[low_mu_loss_balancing.md](low_mu_loss_balancing.md)) fixes training dynamics but not
generalisation: the network learns sharp low-$\mu$ predictions on the training missions yet
predicts the high-$\mu$ mode ($\approx 0.85$) on every visible cell of the validation
missions. A binned cross-entropy $\mu$ head, trained from scratch, collapses identically.
This isolates the binding constraint. It is not the loss shape but cross-mission
generalisation from few low-$\mu$ events (57 in training, 8 in validation). A DINO linear
probe separates low-$\mu$ terrain across missions (mean AUROC $0.827$, leave-one-mission-out),
so the signal is learnable from the images; the model fails to use it. Two levers target the
two remaining causes: too few viewpoints per event, and a mean-seeking regression loss.

## Harvesting: more views per event

Each training window supervises $\mu$ along the next $30\,$s of the driven path. A low-$\mu$
cell is therefore seen from the single trajectory that crossed it. Harvesting adds, per
window, up to $N$ extra supervised cells drawn from the low-$\mu$ states anywhere in the same
mission that fall inside the window's current BEV grid, transformed into the window's anchor
frame and passed through the same augmentation, horizontal flip, and visibility gate as the
path cells. The extra cells carry no temporal discount. Each rare world cell is thus
supervised from every window whose field of view contains it, rather than once. On the three
missions holding most low-$\mu$ terrain this yields on the order of $10^5$ additional low-$\mu$
cell views. Missions without low-$\mu$ events contribute none, and no held-out mission is
touched.

## Balanced MSE

Regression under a squared or smooth-$L_1$ loss is mean-seeking: given an input consistent
with both high and low $\mu$ in the training data, the loss-minimising output is their mean,
which the imbalance pulls toward high $\mu$. Balanced MSE (BMC variant, Ren et al., 2022)
removes this bias without a precomputed density. For a batch of predictions
$\hat{\mu}_i$ and targets $\mu_j$ it minimises

$$
\mathcal{L}_i = -\log \frac{\exp\!\left(-\lVert\hat{\mu}_i - \mu_i\rVert^2 / 2\sigma^2\right)}
{\sum_j \exp\!\left(-\lVert\hat{\mu}_i - \mu_j\rVert^2 / 2\sigma^2\right)},
$$

the cross-entropy between prediction $i$ and its own target against all batch targets under a
learnable noise scale $\sigma$. The batch targets act as a Monte-Carlo sample of the label
distribution, so the balanced training objective matches a uniform test prior. $\sigma$ is
optimised jointly at a higher learning rate. It replaces the smooth-$L_1$ term on $\mu$ only;
$\nu$ keeps its weighted smooth-$L_1$, and evaluation uses smooth-$L_1$ error throughout so
that reported errors remain comparable across runs. BMC supersedes inverse-density weighting,
so LDS is disabled when it is active.

## Experiments

Six configurations were trained, each a delta from the one above. The table isolates the
component varied in each.

| Configuration | $\mu$ loss | Reweighting | Vis. gate | Backbone LR | Harvest |
|---|---|---|---|---|---|
| Baseline | smooth-$L_1$ | oversample $0.4$ + LDS $1.0$ | off | $1{\times}$ | off |
| Mitigated | smooth-$L_1$ | oversample $0.25$ + LDS $0.75$ | **on** | **$0.1{\times}$** | off |
| Low-$\mu$ fine-tune | + asymmetric penalty | oversample $0.4$ | on | $0.02{\times}$ | off |
| Binned CE | **cross-entropy** | oversample $0.25$ + LDS $0.75$ | on | $0.1{\times}$ | off |
| Harvest + BMC | **Balanced MSE** | oversample $0.25$, LDS off | on | $0.1{\times}$ | **$N{=}64$** |

The mitigated configuration is the pivot and changes four things relative to the baseline.
The **visibility gate** zeroes the loss weight of horizon cells outside every input frame's
frustum, which is what exposes the collapse: without it roughly two thirds of the supervised
low-$\mu$ cells are occluded, and the apparent low-$\mu$ skill in the all-cell metric lives
entirely in those cells. The **backbone learning rate** is cut to $0.1\times$ so the depth
cross-entropy does not corrupt the pretrained encoder early, and the **depth loss weight** is
halved from $0.10$ to $0.05$ for the same reason. The **reweighting** is softened (LDS $1.0 \to
0.75$, oversample $0.4 \to 0.25$), and best-checkpoint selection with early stopping is added.
These fix the training dynamics, the loss decreases with a late best epoch and depth MAE falls
to $0.31$, but leave the visible prediction at the mode. Each later configuration then varies
one axis: the fine-tune adds an asymmetric penalty and strong augmentation over a near-frozen
mitigated checkpoint; the binned head replaces the regression loss with classification; harvest
and BMC add viewpoints and replace the loss.

Evaluation uses the predicted $\mu$ against the true $\mu$ per $0.1$-wide bin on visible,
observable validation cells; collapse shows as a near-constant prediction across bins. The
$\mu$ spread (standard deviation of predictions) measures collapse severity directly.

| Configuration | Visible low-$\mu$ prediction | $\mu$ spread | Outcome |
|---|---|---|---|
| Baseline | $\approx 0.84$ | — | Collapse; skill confined to occluded cells |
| Mitigated | flat near mode | $0.24$ | Healthiest dynamics; visible prediction unmoved |
| Low-$\mu$ fine-tune | $\approx 0.82$ | $0.25$ | Flat, unchanged over $15$ epochs |
| Binned CE | $\approx 0.86$ | $0.18$ | Collapse identical; depth destabilised |
| Harvest + BMC | $\approx 0.87$ | $0.078$ | Lowest aggregate error, hardest collapse |

No configuration moves the visible low-$\mu$ prediction off the high-$\mu$ mode. Harvest and
BMC reach the lowest aggregate validation error of any run, but do so by predicting the mode
more tightly: the $\mu$ spread halves to $0.078$ and the visible low-$\mu$ prediction is the
most collapsed of all. Four distinct interventions, distribution reweighting, an asymmetric
penalty, a classification loss, and viewpoint augmentation with a distribution-balanced loss,
fail identically. Since the DINO probe separates low-$\mu$ terrain across missions from the same
images, the signal is present and the failure is not one of separability. The constant
prediction is therefore a cross-mission generalisation limit of the architecture under this
label regime (57 training and 8 validation low-$\mu$ events), not an artefact of the loss, the
output parameterisation, or the number of supervised viewpoints.
