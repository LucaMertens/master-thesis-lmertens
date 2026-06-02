## Camera choice 


We need RGB and depth images, synchronized both in time and space.
The GrandTour dataset has multiple different cameras.
- The Zed2i RGB-D  camera is time-synchronized via USB. The authors of the dataset state that the camera data is therefore not perfectly synchronized to the sensor data. The Zed2i data includes depth information generated in the ROS-driver from the manufacturer (https://github.com/stereolabs/zed-ros-wrapper)
- The alphasense cameras are time-synchronized via PTP (i.e., relatively accurate). 1 RGB camera is mounted between two mono-cameras. Depth information is not available, but could be generated via the NVIDIA-foundation model FoundationStereo.

The Zed2i uses a USB serial buffer for timestamping, which the authors note yields only "unreliable timestamps" with no individual line read-out timings. The CoreResearch (Alphasense) cameras are PTP-synchronized against the NovAtel CPT7 GNSS grandmaster, achieving sub-1 ms alignment with the rest of the sensor suite.
We use the alphasense data due to its better time-sync and generate the depth images using FoundationStereo.


## Why not?

- Why not Zed2i RGB-D? Inaccurate time sync (USB)
- Why not Lidar for depth? The obstacle with LiDAR is density, not resolution. A LiDAR measures a set of 3D points rather than a pixel image. To turn it into a depth image, each point is projected through the camera model onto the image plane and its distance written into the pixel it lands on. Because we choose the target pixel grid ourselves, the points can be rendered at any resolution, so matching resolution is trivial. The real limitation is that a LiDAR returns relatively few points (thin horizontal scan lines), so most pixels of that image stay empty.

  Whether that sparsity matters depends on how the depth is used. The network is not handed depth directly; it learns to estimate depth from the RGB image. The measured depth is only a training signal that teaches this estimate, and the network is corrected only at pixels where a measurement actually exists; pixels with no measurement are simply ignored. So the depth map does not need to be complete; it only has to provide enough correct examples to learn from, which is why sparse LiDAR is usable in principle.

  In practice, though, we train the network from scratch, so its depth estimate starts from nothing and needs a large number of examples to learn well. Stereo provides a depth value at nearly every pixel of every frame, whereas LiDAR's scan lines cover only a small fraction. Stereo's dense supervision is therefore far more effective, so we prefer it.
- Why not depth-only images from Realsense D435 + Alphasense RGB, then project into same frame? Time sync not at the same time, projection issues, cameras are pretty far apart



## Sources

> CoreResearch cameras use IEEE 1588v2 PTP, synchronized against the NovAtel CPT7 GNSS grandmaster, achieving sub-1 ms alignment across the sensor suite.
Grandtour-Paper, to verify

Zed2i
> "connected via a USB serial buffer; therefore, only unreliable timestamps are available and without individual line read-out timings"