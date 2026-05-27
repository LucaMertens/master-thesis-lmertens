

## 1. Dataset

We use the Anymal GrandTour dataset instead of their in-house daataset of a wheeled robot.
For this, we need a few things:

### Camera
RGB and depth images, synchronized both in time and space.
The GrandTour dataset has multiple different cameras.
- The Zed2i RGB-D  camera is time-synchronized via USB. The authors of the dataset state that the camera data is therefore not perfectly synchronized to the sensor data. The Zed2i data includes depth information generated in the ROS-driver from the manufacturer (https://github.com/stereolabs/zed-ros-wrapper)
- The alphasense cameras are time-synchronized via PTP (i.e., relatively accurate). 1 RGB camera is mounted between two mono-cameras. Depth information is not available, but could be generated via the NVIDIA-foundation model FoundationStereo.

The Zed2i uses a USB serial buffer for timestamping, which the authors note yields only "unreliable timestamps" with no individual line read-out timings. The CoreResearch (Alphasense) cameras are PTP-synchronized against the NovAtel CPT7 GNSS grandmaster, achieving sub-1 ms alignment with the rest of the sensor suite.
We use the alphasense data due to its better time-sync and generate the depth images using FoundationStereo.