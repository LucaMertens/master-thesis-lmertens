# GrandTour Dataset — ROS1 Bag Topics

Each mission directory contains a set of bag files. All bags are prefixed with the mission timestamp (e.g. `2024-10-01-11-29-55_`).

---

## `*_alphasense.bag`

### `/boxi/alphasense/front_center/camera_info`
- **Type:** `sensor_msgs/CameraInfo`
- **Frame:** `alphasense_front_center`
- **Fields:**
- `header` (std_msgs__msg__Header)
- `height` (int)
- `width` (int)
- `distortion_model` (str)
- `D` (np.ndarray[tuple[int, ...], np.dtype[np.float64]])
- `K` (np.ndarray[tuple[int, ...], np.dtype[np.float64]])
- `R` (np.ndarray[tuple[int, ...], np.dtype[np.float64]])
- `P` (np.ndarray[tuple[int, ...], np.dtype[np.float64]])
- `binning_x` (int)
- `binning_y` (int)
- `roi` (sensor_msgs__msg__RegionOfInterest)

### `/boxi/alphasense/front_center/image_raw/compressed`
- **Type:** `sensor_msgs/CompressedImage`
- **Frame:** `alphasense_front_center`
- **Fields:**
- `header` (std_msgs__msg__Header)
- `format` (str)
- `data` (np.ndarray[tuple[int, ...], np.dtype[np.uint8]])

### `/boxi/alphasense/front_left/camera_info`
- **Type:** `sensor_msgs/CameraInfo`
- **Frame:** `alphasense_front_left`
- **Fields:**
- `header` (std_msgs__msg__Header)
- `height` (int)
- `width` (int)
- `distortion_model` (str)
- `D` (np.ndarray[tuple[int, ...], np.dtype[np.float64]])
- `K` (np.ndarray[tuple[int, ...], np.dtype[np.float64]])
- `R` (np.ndarray[tuple[int, ...], np.dtype[np.float64]])
- `P` (np.ndarray[tuple[int, ...], np.dtype[np.float64]])
- `binning_x` (int)
- `binning_y` (int)
- `roi` (sensor_msgs__msg__RegionOfInterest)

### `/boxi/alphasense/front_left/image_raw/compressed`
- **Type:** `sensor_msgs/CompressedImage`
- **Frame:** `alphasense_front_left`
- **Fields:**
- `header` (std_msgs__msg__Header)
- `format` (str)
- `data` (np.ndarray[tuple[int, ...], np.dtype[np.uint8]])

### `/boxi/alphasense/front_right/camera_info`
- **Type:** `sensor_msgs/CameraInfo`
- **Frame:** `alphasense_front_right`
- **Fields:**
- `header` (std_msgs__msg__Header)
- `height` (int)
- `width` (int)
- `distortion_model` (str)
- `D` (np.ndarray[tuple[int, ...], np.dtype[np.float64]])
- `K` (np.ndarray[tuple[int, ...], np.dtype[np.float64]])
- `R` (np.ndarray[tuple[int, ...], np.dtype[np.float64]])
- `P` (np.ndarray[tuple[int, ...], np.dtype[np.float64]])
- `binning_x` (int)
- `binning_y` (int)
- `roi` (sensor_msgs__msg__RegionOfInterest)

### `/boxi/alphasense/front_right/image_raw/compressed`
- **Type:** `sensor_msgs/CompressedImage`
- **Frame:** `alphasense_front_right`
- **Fields:**
- `header` (std_msgs__msg__Header)
- `format` (str)
- `data` (np.ndarray[tuple[int, ...], np.dtype[np.uint8]])

### `/boxi/alphasense/left/camera_info`
- **Type:** `sensor_msgs/CameraInfo`
- **Frame:** `alphasense_left`
- **Fields:**
- `header` (std_msgs__msg__Header)
- `height` (int)
- `width` (int)
- `distortion_model` (str)
- `D` (np.ndarray[tuple[int, ...], np.dtype[np.float64]])
- `K` (np.ndarray[tuple[int, ...], np.dtype[np.float64]])
- `R` (np.ndarray[tuple[int, ...], np.dtype[np.float64]])
- `P` (np.ndarray[tuple[int, ...], np.dtype[np.float64]])
- `binning_x` (int)
- `binning_y` (int)
- `roi` (sensor_msgs__msg__RegionOfInterest)

### `/boxi/alphasense/left/image_raw/compressed`
- **Type:** `sensor_msgs/CompressedImage`
- **Frame:** `alphasense_left`
- **Fields:**
- `header` (std_msgs__msg__Header)
- `format` (str)
- `data` (np.ndarray[tuple[int, ...], np.dtype[np.uint8]])

### `/boxi/alphasense/right/camera_info`
- **Type:** `sensor_msgs/CameraInfo`
- **Frame:** `alphasense_right`
- **Fields:**
- `header` (std_msgs__msg__Header)
- `height` (int)
- `width` (int)
- `distortion_model` (str)
- `D` (np.ndarray[tuple[int, ...], np.dtype[np.float64]])
- `K` (np.ndarray[tuple[int, ...], np.dtype[np.float64]])
- `R` (np.ndarray[tuple[int, ...], np.dtype[np.float64]])
- `P` (np.ndarray[tuple[int, ...], np.dtype[np.float64]])
- `binning_x` (int)
- `binning_y` (int)
- `roi` (sensor_msgs__msg__RegionOfInterest)

### `/boxi/alphasense/right/image_raw/compressed`
- **Type:** `sensor_msgs/CompressedImage`
- **Frame:** `alphasense_right`
- **Fields:**
- `header` (std_msgs__msg__Header)
- `format` (str)
- `data` (np.ndarray[tuple[int, ...], np.dtype[np.uint8]])

---

## `*_alphasense_imu.bag`

### `/boxi/alphasense/imu`
- **Type:** `sensor_msgs/Imu`
- **Frame:** `alphasense_imu`
- **Fields:**
- `header` (std_msgs__msg__Header)
- `orientation` (geometry_msgs__msg__Quaternion)
- `orientation_covariance` (np.ndarray[tuple[int, ...], np.dtype[np.float64]])
- `angular_velocity` (geometry_msgs__msg__Vector3)
- `angular_velocity_covariance` (np.ndarray[tuple[int, ...], np.dtype[np.float64]])
- `linear_acceleration` (geometry_msgs__msg__Vector3)
- `linear_acceleration_covariance` (np.ndarray[tuple[int, ...], np.dtype[np.float64]])

---

## `*_anymal_command.bag`

### `/anymal/twist_command`
- **Type:** `geometry_msgs/TwistStamped`
- **Frame:** `N/A`
- **Fields:**
- `header` (std_msgs__msg__Header)
- `twist` (geometry_msgs__msg__Twist)

---

## `*_cpt7_ie_tc.bag`

### `/boxi/inertial_explorer/tc/cpt7_imu_acc_bias`
- **Type:** `geometry_msgs/Vector3Stamped`
- **Frame:** `enu_origin`
- **Fields:**
- `header` (std_msgs__msg__Header)
- `vector` (geometry_msgs__msg__Vector3)

### `/boxi/inertial_explorer/tc/cpt7_imu_gyro_drift`
- **Type:** `geometry_msgs/Vector3Stamped`
- **Frame:** `enu_origin`
- **Fields:**
- `header` (std_msgs__msg__Header)
- `vector` (geometry_msgs__msg__Vector3)

### `/boxi/inertial_explorer/tc/navsatfix`
- **Type:** `sensor_msgs/NavSatFix`
- **Frame:** `enu_origin`
- **Fields:**
- `header` (std_msgs__msg__Header)
- `status` (sensor_msgs__msg__NavSatStatus)
- `latitude` (float)
- `longitude` (float)
- `altitude` (float)
- `position_covariance` (np.ndarray[tuple[int, ...], np.dtype[np.float64]])
- `position_covariance_type` (int)

### `/boxi/inertial_explorer/tc/odometry`
- **Type:** `nav_msgs/Odometry`
- **Frame:** `enu_origin → cpt7_imu`
- **Fields:**
- `header` (std_msgs__msg__Header)
- `child_frame_id` (str)
- `pose` (geometry_msgs__msg__PoseWithCovariance)
- `twist` (geometry_msgs__msg__TwistWithCovariance)

### `/boxi/inertial_explorer/tc/raw`
- **Type:** `gnss_msgs/GnssRaw`
- **Frame:** `(unregistered type)`
- **Fields:**
- *(type not registered)*

### `/tf`
- **Type:** `tf2_msgs/TFMessage`
- **Transform pairs (parent → child):**
  - `cpt7_imu` → `enu_origin`

---

## `*_tf_minimal.bag`

### `/tf`
- **Type:** `tf2_msgs/TFMessage`
- **Transform pairs (parent → child):**
  - `LF_HAA_drive` → `LF_HIP`
  - `LF_HFE_output` → `LF_THIGH`
  - `LF_KFE_drive` → `LF_SHANK`
  - `LH_HAA_drive` → `LH_HIP`
  - `LH_HFE_output` → `LH_THIGH`
  - `LH_KFE_drive` → `LH_SHANK`
  - `RF_HAA_drive` → `RF_HIP`
  - `RF_HFE_output` → `RF_THIGH`
  - `RF_KFE_drive` → `RF_SHANK`
  - `RH_HAA_drive` → `RH_HIP`
  - `RH_HFE_output` → `RH_THIGH`
  - `RH_KFE_drive` → `RH_SHANK`
  - `base` → `odom`
  - `cpt7_imu` → `enu_origin`
  - `hesai_lidar` → `dlio_map`
  - `hesai_lidar` → `dlio_odom`

### `/tf_static`
- **Type:** `tf2_msgs/TFMessage`
- **Transform pairs (parent → child):**
  - `LF_HFE_drive` → `LF_thigh_fixed`
  - `LF_HIP` → `LF_hip_fixed`
  - `LF_SHANK` → `LF_shank_fixed`
  - `LF_THIGH` → `LF_HFE_drive`
  - `LF_hip_fixed` → `LF_HFE_output`
  - `LF_shank_fixed` → `LF_FOOT`
  - `LF_thigh_fixed` → `LF_KFE_drive`
  - `LH_HFE_drive` → `LH_thigh_fixed`
  - `LH_HIP` → `LH_hip_fixed`
  - `LH_SHANK` → `LH_shank_fixed`
  - `LH_THIGH` → `LH_HFE_drive`
  - `LH_hip_fixed` → `LH_HFE_output`
  - `LH_shank_fixed` → `LH_FOOT`
  - `LH_thigh_fixed` → `LH_KFE_drive`
  - `RF_HFE_drive` → `RF_thigh_fixed`
  - `RF_HIP` → `RF_hip_fixed`
  - `RF_SHANK` → `RF_shank_fixed`
  - `RF_THIGH` → `RF_HFE_drive`
  - `RF_hip_fixed` → `RF_HFE_output`
  - `RF_shank_fixed` → `RF_FOOT`
  - `RF_thigh_fixed` → `RF_KFE_drive`
  - `RH_HFE_drive` → `RH_thigh_fixed`
  - `RH_HIP` → `RH_hip_fixed`
  - `RH_SHANK` → `RH_shank_fixed`
  - `RH_THIGH` → `RH_HFE_drive`
  - `RH_hip_fixed` → `RH_HFE_output`
  - `RH_shank_fixed` → `RH_FOOT`
  - `RH_thigh_fixed` → `RH_KFE_drive`
  - `alphasense_base` → `alphasense_front_center`
  - `alphasense_base` → `alphasense_front_left`
  - `alphasense_base` → `alphasense_front_right`
  - `alphasense_base` → `alphasense_left`
  - `alphasense_base` → `alphasense_right`
  - `base` → `LF_HAA_drive`
  - `base` → `LH_HAA_drive`
  - `base` → `RF_HAA_drive`
  - `base` → `RH_HAA_drive`
  - `base` → `anymal_imu`
  - `base` → `box_base`
  - `base` → `depth_camera_front_lower`
  - `base` → `depth_camera_front_upper`
  - `base` → `depth_camera_left`
  - `base` → `depth_camera_rear_lower`
  - `base` → `depth_camera_rear_upper`
  - `base` → `depth_camera_right`
  - `base` → `velodyne_lidar`
  - `box_base` → `adis16475_imu`
  - `box_base` → `alphasense_base`
  - `box_base` → `alphasense_imu`
  - `box_base` → `ap20_imu`
  - `box_base` → `cpt7_imu`
  - `box_base` → `hdr_base`
  - `box_base` → `hesai_lidar`
  - `box_base` → `livox_lidar`
  - `box_base` → `prism`
  - `box_base` → `stim320_imu`
  - `box_base` → `zed2i_base`
  - `hdr_base` → `hdr_front`
  - `hdr_base` → `hdr_left`
  - `hdr_base` → `hdr_right`
  - `livox_lidar` → `livox_imu`
  - `zed2i_base` → `zed2i_imu`
  - `zed2i_base` → `zed2i_left`
  - `zed2i_base` → `zed2i_right`

---

## `*_zed2i_images.bag`

### `/boxi/zed2i/left/camera_info`
- **Type:** `sensor_msgs/CameraInfo`
- **Frame:** `zed2i_left`
- **Fields:**
- `header` (std_msgs__msg__Header)
- `height` (int)
- `width` (int)
- `distortion_model` (str)
- `D` (np.ndarray[tuple[int, ...], np.dtype[np.float64]])
- `K` (np.ndarray[tuple[int, ...], np.dtype[np.float64]])
- `R` (np.ndarray[tuple[int, ...], np.dtype[np.float64]])
- `P` (np.ndarray[tuple[int, ...], np.dtype[np.float64]])
- `binning_x` (int)
- `binning_y` (int)
- `roi` (sensor_msgs__msg__RegionOfInterest)

### `/boxi/zed2i/left/image_raw/compressed`
- **Type:** `sensor_msgs/CompressedImage`
- **Frame:** `zed2i_left`
- **Fields:**
- `header` (std_msgs__msg__Header)
- `format` (str)
- `data` (np.ndarray[tuple[int, ...], np.dtype[np.uint8]])

### `/boxi/zed2i/right/camera_info`
- **Type:** `sensor_msgs/CameraInfo`
- **Frame:** `zed2i_right`
- **Fields:**
- `header` (std_msgs__msg__Header)
- `height` (int)
- `width` (int)
- `distortion_model` (str)
- `D` (np.ndarray[tuple[int, ...], np.dtype[np.float64]])
- `K` (np.ndarray[tuple[int, ...], np.dtype[np.float64]])
- `R` (np.ndarray[tuple[int, ...], np.dtype[np.float64]])
- `P` (np.ndarray[tuple[int, ...], np.dtype[np.float64]])
- `binning_x` (int)
- `binning_y` (int)
- `roi` (sensor_msgs__msg__RegionOfInterest)

### `/boxi/zed2i/right/image_raw/compressed`
- **Type:** `sensor_msgs/CompressedImage`
- **Frame:** `zed2i_right`
- **Fields:**
- `header` (std_msgs__msg__Header)
- `format` (str)
- `data` (np.ndarray[tuple[int, ...], np.dtype[np.uint8]])
