# simple_labeling_pipeline_frey-et-al

Baseline velocity-error labels after Frey et al., "Fast Traversability Estimation for Wild Visual Navigation", 2023. `labeling.py` implements it, `hyperparameters.py` holds the settings.

- The Inertial Explorer velocity in `*_cpt7_ie_tc.bag` is unusable (zero angular twist, linear velocity in NED while pose is ENU, global instead of local frame). Velocity is obtained by differentiating the odometry pose.
