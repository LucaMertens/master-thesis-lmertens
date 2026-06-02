FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=US/Pacific
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# System libs — many are runtime deps of opencv-contrib and open3d, keep them.
RUN apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends \
        apt-utils bzip2 build-essential ca-certificates checkinstall cmake curl \
        ffmpeg freeglut3-dev g++ gcc gfortran git libatlas-base-dev libavcodec-dev \
        libavformat-dev libblas-dev libccd-dev libcgal-dev libfaac-dev libflann-dev \
        libglew-dev libgphoto2-dev libgtk2.0-dev libhdf5-dev libjpeg8-dev liblapack-dev \
        libmp3lame-dev libopencore-amrnb-dev libopencore-amrwb-dev libproj-dev \
        libswscale-dev libtbb-dev libtheora-dev libtiff5-dev libturbojpeg-dev \
        libusb-1.0-0-dev libv4l-dev libvorbis-dev libxine2-dev libxvidcore-dev \
        libyaml-cpp-dev libzmq3-dev openexr p7zip-full pkg-config proj-data \
        protobuf-compiler software-properties-common wget x264 xvfb yasm zlib1g-dev && \
    rm -rf /var/lib/apt/lists/*

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /miniconda.sh && \
    /bin/bash /miniconda.sh -b -p /opt/conda && \
    rm /miniconda.sh && \
    /opt/conda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    /opt/conda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && \
    /opt/conda/bin/conda create -n foundation_stereo python=3.11 -y && \
    /opt/conda/bin/conda clean -a -y

# PATH must point at the env — exec-form ENTRYPOINT doesn't run a login shell.
ENV PATH=/opt/conda/envs/foundation_stereo/bin:$PATH
ENV OPENCV_IO_ENABLE_OPENEXR=1

# Torch from the cu124 index (verbatim from FoundationStereo's own Dockerfile).
RUN pip install --no-cache-dir torch==2.4.0 torchvision==0.19.0 \
        --index-url https://download.pytorch.org/whl/cu124

# flash-attn pinned to 2.6.3 — newer releases need torch >= 2.5.
# xformers from cu124 to match torch. Order matters: torch first, then flash-attn.
ARG MAX_JOBS=4
ENV MAX_JOBS=${MAX_JOBS}
RUN pip install --no-cache-dir \
        scikit-image omegaconf opencv-contrib-python imgaug Ninja timm albumentations \
        scipy joblib scikit-learn ruamel.yaml trimesh pyyaml imageio open3d \
        transformations einops gdown tqdm && \
    pip install --no-cache-dir flash-attn==2.6.3 --no-build-isolation && \
    pip install --no-cache-dir xformers==0.0.28.post1 \
        --index-url https://download.pytorch.org/whl/cu124

ARG FOUNDATIONSTEREO_COMMIT=6e8806816b533e4d13ddbb95ffa907b797060a62
RUN git clone https://github.com/NVlabs/FoundationStereo /app/FoundationStereo && \
    git -C /app/FoundationStereo checkout ${FOUNDATIONSTEREO_COMMIT}

WORKDIR /app
COPY generate_depth.py /app/generate_depth.py

ENTRYPOINT ["python", "generate_depth.py"]
CMD ["--ckpt", "/app/pretrained_models/23-51-11/model_best_bp2.pth", "--missions", "/data/ETH-1"]
