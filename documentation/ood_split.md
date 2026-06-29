# Out-of-Distribution Split


## Split

Counts are timesteps at 10 Hz; `low-mu` is the confident-low-traversability count
(mu <= 0.75 at observability >= 0.6), the scarce safety-critical class.

| Set | Runs | frames | low-mu |
|---|---|---|---|
| ID train | ALB-1, ALB-2, CYN-1, HOEB-1, LMB-1, RIV-1, TRIM-1, HIL-1, KAEB-2, KAEB-3 | 41254 | 3810 |
| ID held-out (val = test) | ALB-3, LMB-2, HOEB-2, CYN-2 | 11544 | 1064 |
| near-OOD: forest | KAEB-1, ROOT-1 | 7083 | 1248 |
| near-OOD: gravel/mountain | EIG-1/2, PIL-2 | 11137 | 1264 |
| far-OOD: SNOW | SNOW-1/2/3 | 5622 | 462 |
| far-OOD: ARC | ARC-1/2/3/4 | 17507 | 2254 |

Train and validation are merged into a single training set. The held-out ID set serves
three roles: it drives checkpoint selection, provides the RQ1 in-domain test, and acts
as the negative class for OOD detection. Sensor dropout is this same held-out set with
the depth input zeroed, not a separate set.

## Rationale

### In-distribution domain

The ID domain only consists of forest-like environments: dirt, leaves, roots, gravel trails, and vegetation.
The narrow definition is deliberate and exists to give a clean OOD boundary. Extending
ID to campus/urban would introduce concrete, pavement, stairs, and indoor scenes
into the distribution, at which point almost every remaining environment overlaps it and only
"snow" stays clearly outside. Restricting ID to forest keeps the "built" and "snow"
environments genuinely far, and lets the near-OOD shifts (KAEB-1, ROOT-1) be measured
against a pure-forest reference.

The cost is data volume: the forest-only training set holds ~41k frames (~69 min of
distinct traversal), including the two ordinary Kaeferberg runs KAEB-2/3. The
ImageNet-pretrained ResNet-34 backbone, together with flip and colour augmentation and
low-mu oversampling, should keep this trainable.

### ID-internal split

The two ID sets are split by whole run but drawn from the same families, so the same
terrain appears in both. This repetition is intentional: the held-out set is the OOD
negative class and must therefore be unambiguously in-distribution. Unseen locations
would undermine that, the opposite of what a generalization benchmark needs. Every ID
family is represented in training, and both sets retain confident-low-mu samples
(train 3810, held-out 1064); without them the in-domain low-traversability error
cannot be measured. ROOT-1, previously a training run, now serves as near-OOD.

The Kaeferberg (KAEB) family deliberately straddles the ID/OOD boundary: KAEB-2/3 sit
in training as ordinary forest, while KAEB-1 is held out as near-OOD. The split stays
disjoint at the run level, so no run appears on both sides; KAEB is the only family
allowed to span ID and OOD, and it does so to make KAEB-1 a within-environment shift
(see Near-OOD).

Using the held-out set for checkpoint selection affects only the RQ1 number. AUROC and
FPR95 are computed afterwards and remain unaffected, and the architecture is fixed, so
no hyperparameter-search bias accumulates.

### Near-OOD

Near-OOD spans a graded set of small shifts from the forest ID, in two tiers.

The **forest tier** (KAEB-1, ROOT-1) holds forest runs each displaced from the ID
forest along one axis. KAEB-1 shifts the vegetation (dense, entangling undergrowth,
where mu collapses to 0.0 as the robot snags). ROOT-1 shifts the geometry (large roots
and boulders, barely present in ID training). The two axes are independent, so the tier
probes two distinct subtle shifts. The shift is meant to be small: training already
contains related cases (ALB-2 ground-cover, RIV-1 tree trunks), so a detector should
find these hard, which is the point. KAEB-1 is specifically a within-environment
near-OOD: KAEB-2/3 place the same Kaeferberg forest in ID, so a detector must respond to
the entanglement hazard, not the location. The tier totals 7083 frames (~470 roughly
independent scenes), enough for a stable FPR95.

The **gravel/mountain tier** (EIG-1/2, PIL-2) holds alpine and mountain runs whose
underground stays gravel and dirt, shared with the ID forest trails (CYN, HIL, TRIM).
They are near on the surface axis, but the scene is displaced (open alpine terrain,
stone and metal-grating stairs), so they sit between the forest tier and the far sets
and give a graded near->far difficulty. PIL-1 is excluded because its surface is stone
stairs and cobblestone rather than gravel. The tier carries strong low-mu (1264, much of
it on EIG's metal-mesh stairs), useful for the failure-mode analysis, and at 11137
frames is comfortably stable.

### Far-OOD

Far-OOD combines two contrasting environments, both clearly outside a forest ID, each
evaluated separately. SNOW is a surface shift (packed snow). ARC is a
built/industrial shift (concrete, metal-grating stairs with gaps, half-demolished
buildings, indoor and outdoor). The forest-only ID is what lets the built shift count
as far: against the broader ID tried earlier, ARC and the construction site CON shared
too much gravel, concrete, and stairs.

Only the four day runs ARC-1..4 are used (17507 frames, 2254 low-mu), the highest
low-mu count of any far set. This tests whether the network mispredicts confidently on
dangerous, unfamiliar terrain. SNOW is smaller (5622 frames, ~370 independent scenes),
so its FPR95 is the noisier of the two, while AUROC stays stable. ARC is large enough
to be stable on both metrics.

### Sensor dropout

Sensor dropout is not a separate environment but the held-out ID set with the depth
input zeroed. Because training applies 30% point-cloud dropout, the RGB-only path
remains functional, and this pass tests whether a detector responds to the missing
modality. It also provides the RGB-only branch for the modality-disagreement ensemble
(optional RQ5).

### Exclusions

The campus/urban families (ETH, GRI, LEICA, LEE) and the museum (HAUS) are excluded
from ID, since they are built or paved scenes that would blur the boundary; they are
not used as OOD either, HAUS in particular being a flat indoor floor with almost no
low-mu. PIL-1 stays out of the gravel near-OOD tier because its surface is stone stairs
and cobblestone, not gravel. For ARC, the dawn run (ARC-5) and the smoke/night runs
(ARC-6/7) are left out so that the far set captures an environment shift rather than a
lighting one; depth dropout already covers the non-environment case. The construction
site (CON), the train tracks (SBB), and the remaining mountain runs (SPX, HEAP) stay
unused: SNOW and ARC already cover the far axes and EIG/PIL-2 cover the gravel near
tier. The red-tint correction (see `red_tint_correction.md`) fixes the heavily affected
runs (ARC-5/6/7, SPX-3), which are excluded here regardless; ARC-1..4 retain only brief
tint on a few indoor frames.
