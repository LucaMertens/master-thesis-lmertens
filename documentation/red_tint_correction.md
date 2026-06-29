## Red-tint correction

In dark scenes the front-center Alphasense camera occasionally fails its
auto-white-balance and bakes a strong red cast into the recorded frames. The
effect is per-frame and is present in the raw bags, so it cannot be undone by a
fixed color transform. It affects several missions (ARC-5/6/7, SPX-3, and parts
of the ARC and CON sequences) and would otherwise present the network with a
spurious, mission-correlated color shift.

The correction exploits the fact that within an affected mission most frames are
correctly balanced. We flag a frame as tinted when its redness ratio,
$R / \tfrac{1}{2}(G + B)$ over the valid (non-border) pixels, exceeds 1.5. The
clean frames define the true color balance: their median per-channel chroma,
computed over a local time window around each tinted frame, gives a reference
white point. Each tinted frame is rescaled per channel onto that reference and
its brightness matched to the clean frames. The transform reuses the mission's
own correct exposure rather than imposing a neutral gray-world assumption, which
should keep the corrected frames close to the correct color distribution of the
dataset.

![Red-tinted frames (left) and the corrected output (right) for two affected
missions.](figures/tint_before_after.png)

The corrected frames recover the neutral, low-light appearance of the
surrounding clean frames while preserving scene structure. The remaining bluish
cast reflects the genuine dusk illumination of these sequences rather than a
residual artifact.

Further before/after pairs (tinted left, corrected right):

![ARC-1: red-tinted (left) and corrected (right).](figures/tint_before_after_ARC-1.png)

![ARC-3: red-tinted (left) and corrected (right).](figures/tint_before_after_ARC-3.png)

![ARC-7: red-tinted (left) and corrected (right).](figures/tint_before_after_ARC-7.png)
