## Camera choice 


We RGB and depth images, synchronized both in time and space.
The GrandTour dataset has multiple different cameras.
- The Zed2i RGB-D  camera is time-synchronized via USB. The authors of the dataset state that the camera data is therefore not perfectly synchronized to the sensor data. The Zed2i data includes depth information generated in the ROS-driver from the manufacturer (https://github.com/stereolabs/zed-ros-wrapper)
- The alphasense cameras are time-synchronized via PTP (i.e., relatively accurate). 1 RGB camera is mounted between two mono-cameras. Depth information is not available, but could be generated via the NVIDIA-foundation model FoundationStereo.

The Zed2i uses a USB serial buffer for timestamping, which the authors note yields only "unreliable timestamps" with no individual line read-out timings. The CoreResearch (Alphasense) cameras are PTP-synchronized against the NovAtel CPT7 GNSS grandmaster, achieving sub-1 ms alignment with the rest of the sensor suite.
We use the alphasense data due to its better time-sync and generate the depth images using FoundationStereo.


## Why not?

- Why not Zed2i RGB-D? Inaccurate time sync (USB)
- Why not Lidar for depth? Not the same density, would need interpolation
- Why not depth-only images from Realsense D435 + Alphasense RGB, then project into same frame? Time sync not at the same time, projection issues, cameras are pretty far apart



## Sources

> CoreResearch cameras use IEEE 1588v2 PTP, synchronized against the NovAtel CPT7 GNSS grandmaster, achieving sub-1 ms alignment across the sensor suite.
Grandtour-Paper, to verify

Zed2i
> "connected via a USB serial buffer; therefore, only unreliable timestamps are available and without individual line read-out timings"