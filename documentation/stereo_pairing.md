# Stereo pairing for RGB–depth generation

The network is trained on RGB–depth pairs that must be aligned in space: every
colour pixel needs a depth value at the same image coordinate. The Alphasense
front module provides three rigidly mounted, time-synchronised cameras in a
horizontal row — a monochrome stereo pair (`front_left`, `front_right`) with a
colour camera (`front_center`) at its exact midpoint:

| Pair | Baseline | Cameras | Exposure |
|------|---------:|---------|----------|
| left ↔ right | 110 mm | mono + mono | synchronised (built for stereo) |
| center ↔ right | 55 mm | colour + mono | center auto-exposed, not synced |

Depth is estimated with [FoundationStereo](https://github.com/NVlabs/FoundationStereo).
The colour image must come from `front_center`, so the question is which pair to
feed the stereo network.

## First attempt: the monochrome pair, reprojected

The natural choice is the dedicated `front_left`/`front_right` pair: same
modality, exposure-synchronised, and the full 110 mm baseline. We rectified the
pair, ran FoundationStereo, and obtained a disparity map in the rectified-left
frame. Because the colour image lives in the `front_center` frame, the resulting
depth was then reprojected — back-projected to a point cloud and forward-warped
into the center camera with a z-buffer — so it could be paired with the RGB
frame.

"Forward-warping" means we take each pixel of the source depth image, compute
where its 3D point lands in the target (center) camera, and write the depth value
there — we *push* pixels from source to target. The catch is that this mapping is
not one-to-one: as the grid of source pixels is reshaped by the change of
viewpoint, several pixels can land on the same target pixel (resolved by keeping
the nearest one, the z-buffer) while other target pixels receive nothing at all
and are left empty. (The alternative, *backward*-warping — for each target pixel,
look up where it came from in the source — avoids empty pixels but needs the
target depth in advance, which is exactly what we are trying to compute.)

This produced clean disparity but two reprojection artifacts in the warped depth:

- A fixed lattice of empty pixels: Forward-warping a dense pixel grid into a
  slightly different viewpoint stretches it by a few percent (more towards the
  image edges), so a thin, regularly-spaced set of target pixels receives no
  source point and stays empty. Because the camera geometry is static, the same
  hole pattern appears in *every* frame.
- Holes and ghosting along object edges: The 55 mm parallax between the
  left and center viewpoints disoccludes a strip of background beside every near
  object (e.g. a pole shifts ~30 px against the far background). That strip is
  invisible to the left camera, so it has no depth, and stereo half-occlusion
  errors pile up on the opposite edge. Thin objects end up flanked by empty and
  spurious depth.

The lattice is cosmetic and could be hole-filled, but the edge disocclusion is
inherent to warping depth between two viewpoints and cannot be removed in
post-processing, since the information is simply not there.

## Chosen setup: center + right

We instead pair the colour `front_center`, used as the stereo *reference*, with
the monochrome `front_right`, and estimate disparity directly in the center frame.
The depth is then **natively aligned** with the colour image: there is no
reprojection, hence no lattice and no disocclusion holes, and the RGB–depth
correspondence is exact by construction. It also removes a separate
timestamp-matching step, since depth and colour now originate from the same frame.

We deliberately pair with `front_right` rather than `front_left`. Stereo networks
only produce positive disparity, which requires the reference to be the left-hand
camera of the pair; `front_center` sits to the left of `front_right` but to the
right of `front_left`. Pairing with `front_right` therefore lets the network run
in its natural order, whereas `front_left` would require mirroring both images. We
confirmed this on the rectified pair: matched features share the same image row
(vertical offset ≈ 0 px) and have positive horizontal disparity. The two choices
have the same 55 mm baseline, so nothing is lost.

**Upsides**

- No reprojection artifacts; RGB and depth are pixel-aligned by construction.
- The colour image is retained, which matters for terrain semantics (grass vs.
  gravel vs. asphalt) and suits the ImageNet-pretrained RGB backbone.

**Downsides**

- *Halved baseline (55 mm vs. 110 mm).* Stereo depth uncertainty scales with
  `z² / (f · b)`, so halving the baseline roughly doubles the depth error. In the
  near field that the robot actually traverses this stays small (a few cm at 2 m,
  ~25 cm at 4 m) and is acceptable; far-range depth degrades, but stereo is
  unreliable there regardless.
- *Cross-modal, unsynchronised matching.* The monochrome pair was synchronised
  specifically to ease stereo matching; pairing it with the auto-exposed colour
  camera reintroduces a photometric mismatch (different gain, exposure and
  spectral response) that can cause matching errors in high-dynamic-range scenes.
  Timing is not an issue — all cameras are triggered simultaneously and stamped at
  the exposure midpoint — so the mismatch is purely in appearance, which the
  learned matcher tolerates reasonably well.

In short, we trade a modest, near-field-tolerable loss of depth precision and a
harder matching problem for the structural elimination of the reprojection
artifacts and exact RGB–depth alignment.
