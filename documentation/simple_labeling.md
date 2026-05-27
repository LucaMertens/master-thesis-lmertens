<fieldset>
<legend>
Excerpt from
Frey, et al., “Fast Traversability Estimation for Wild Visual Navigation,” 2023.
</legend>

$$
v_{\text {error }}=\frac{1}{2}\left(\left(\bar{v}_x-v_x\right)^2+\left(\bar{v}_y-v_y\right)^2\right)
$$

As $v_{\text {error }}$ will be a noisy scalar signal, we smooth it with a 1 -D Kalman Filter before passing it through a sigmoid function to obtain a valid traversability score:

$$
\tau=\operatorname{sigmoid}\left(-k\left(v_{\text {error }}-v_{\text {thr }}\right)\right)
$$

with $k$ the steepness of the sigmoid, and $v_{\text {thr }}$ the midpoint of the sigmoid that assigns a traversability score of 0.5 . These values can be calibrated depending on the motion specifications of each platform and determine how the velocity error is stretched to the $[0,1]$ interval.

</fieldset>




## Data source
The (`*_cpt7_ie_tc.bag` > `/boxi/inertial_explorer/tc/odometry.twist`) has some peculiarities:
First, the `twist.angular` field is always zero (most likely not populated by the Inertial Explorer).
Second:
While the pose_orien is correctly published in the ROS-standard ENU (East-North-Up) frame, the twist_lin (velocity) is mistakenly left in the sensor's native NED (North-East-Down) frame. Furthermore, the IE velocity is global, while the SE velocity is local.

To perfectly align cpt7_ie_tc_odometry with anymal_state_state_estimator, you need to:

1. Fix the velocity frame: Map the raw NED velocity to ENU.
2. Global to Local: Rotate the corrected ENU velocity into the local IMU frame using the inverse of pose_orien.
3. Apply Extrinsics: Apply the static TF from the IMU to the Robot Base (a 180-degree rotation around the X-axis).


Therefore, we use the numerical derrivatives of the odometry `pose` and `orien` as a data basis.

