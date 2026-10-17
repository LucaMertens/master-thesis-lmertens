# Issues with the WayFASTER codebase



## Euler convention mix-up in the extrinsics composition 

Euler angles decompose a rotation into three elemental rotations, but the same three axis letters have two different conventions (written as `xyz` vs `XYZ` in scipy).
With intrinsic angles, each rotation is about an axis of the body frame as carried along by the previous ones; yaw-pitch-roll is the intrinsic version, and it splits cleanly into a heading rotation and a body tilt independent of that heading.
With extrinsic angles, all three rotations are about the fixed world axes instead; the resulting angles mix heading and tilt together, so the same physical nose-up tilt is reported as a different number depending on which way the robot is facing.

WayFASTER's data loader requires the intrinsic split: a heading rotation defining the BEV anchor frame, and a tilt applied to the camera mount inside it. Instead it decomposes the orientation using the *extrinsic* convention and recomposes the resulting angles as if they were intrinsic pitch and roll. Because the extrinsic angles already have heading baked in, the tilt ends up applied about axes that are wrong by the heading angle.

This likely went unnoticed upstream because WayFASTER's dataset was recorded by a wheeled robot on mostly flat terrain, where pitch and roll stay near zero and the two conventions coincide.


## Ground Height Estimation from Leg Kinematics


All projections between the bird's-eye-view grid and the camera image assumed
the ground to lie at z = 0 in the anchor frame.
This frame originates at the robot's `base` link, which on ANYmal sits roughly 0.55 m above the terrain while walking, dependent on the leg states.
Every projected quantity, including BEV cells, was therefore placed half a meter above the actual ground.
The original WayFASTER system did not exhibit this problem because its wheeled platform carries the base frame close to the ground.
To obtain the height of the robot base, we calculate the mean height of the two lowest feet as an approximation.
Missions without the kinematic data fall back to a mean value of −0.55 m.