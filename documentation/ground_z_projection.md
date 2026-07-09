# Ground Height Estimation from Leg Kinematics


All projections between the bird's-eye-view grid and the camera image assumed
the ground to lie at z = 0 in the anchor frame.
This frame originates at the robot's *base* link, which on ANYmal sits roughly 0.55 m above the terrain while walking, dependent on the leg states.
Every projected quantity, including BEV cells, was therefore placed half a meter above the actual ground.
The original WayFASTER system did not exhibit this problem because its wheeled platform carries the base frame close to the ground.
To obtain the height of the robot base, we calculate the mean height of the two lowest feet as an approximation.
Missions without the kinematic data fall back to a mean value of −0.55 m.