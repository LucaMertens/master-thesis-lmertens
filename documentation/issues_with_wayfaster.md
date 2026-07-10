# Issues inherited from the WayFASTER codebase

Bugs that exist in the upstream [WayFASTER](https://github.com/matval/wayfaster) implementation and that we inherited by adapting it.

## Euler convention mix-up in the extrinsics composition (fixed 2026-07-10)

Euler angles decompose a rotation into three elemental rotations, but the same three axis letters have two different conventions (written as `xyz` vs `XYZ` in scipy). With **intrinsic** angles, each rotation is about an axis of the body frame as carried along by the previous ones; yaw-pitch-roll is the intrinsic version, and it splits cleanly into a heading rotation and a body tilt independent of that heading. With **extrinsic** angles, all three rotations are about the fixed world axes instead; the resulting angles mix heading and tilt together, so the same physical nose-up tilt is reported as a different number depending on which way the robot is facing.

WayFASTER's data loader requires the intrinsic split: a heading rotation defining the BEV anchor frame, and a tilt applied to the camera mount inside it. Instead it decomposes the orientation using the *extrinsic* convention and recomposes the resulting angles as if they were intrinsic pitch and roll. Because the extrinsic angles already have heading baked in, the tilt ends up applied about axes that are wrong by the heading angle, with an error growing to twice the true tilt at opposite headings. The problem was fixed by decomposing and recomposing consistently in the intrinsic convention.

This went unnoticed upstream because WayFASTER's dataset was recorded by a wheeled robot on mostly flat terrain, where pitch and roll stay near zero and the two conventions coincide. Averaged over our training missions, the resulting camera misorientation was $8.0^\circ$, exceeding $10^\circ$ for 29% of states and $68^\circ$ in the worst case, corrupting the image-to-BEV projection, depth voxelization, and visibility gating on sloped terrain.
