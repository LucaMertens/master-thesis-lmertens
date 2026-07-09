# Note on WayFASTER's Training Configuration

The published WayFASTER configuration trains with batch size 4 per GPU and disables pose augmentation (translation and rotation set to zero), while photometric jitter and horizontal flip remain active.
The batch size reflects the memory footprint of lift-splat BEV models, where each sample carries several temporal frames and a full depth-frustum feature volume; comparable architectures train in the same range.
The disabled pose augmentation is unexplained by the authors, but two observations offer a likely cause: with a forward-biased grid and camera, rotations up to ±90° move most of the supervised path out of both the grid and the field of view, and the upstream implementation composes flip and pose augmentation inconsistently, corrupting the label geometry of every flipped sample once pose augmentation is non-zero.
With pose augmentation at zero this composition bug becomes inert, so the published configuration sidesteps a defect that resurfaced, and was fixed, in this work.
