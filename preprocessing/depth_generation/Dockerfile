FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Runtime deps: python, git for the clone, GL/gomp/usb for opencv & open3d.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3-pip git ca-certificates \
    libgl1 libglib2.0-0 libgomp1 libegl1 libusb-1.0-0 && \
    rm -rf /var/lib/apt/lists/*

ENV OPENCV_IO_ENABLE_OPENEXR=1

# Torch from the cu124 index
RUN pip install --no-cache-dir torch==2.4.0 torchvision==0.19.0 \
    --index-url https://download.pytorch.org/whl/cu124 && \
    pip install --no-cache-dir xformers==0.0.28.post1 \
    --index-url https://download.pytorch.org/whl/cu124 && \
    pip install --no-cache-dir \
    opencv-python-headless scipy omegaconf timm einops tqdm \
    imageio joblib trimesh open3d pandas

# Shallow fetch of the pinned commit; full history is >1GB.
ARG FOUNDATIONSTEREO_COMMIT=6e8806816b533e4d13ddbb95ffa907b797060a62
RUN git init /app/FoundationStereo && \
    git -C /app/FoundationStereo remote add origin https://github.com/NVlabs/FoundationStereo && \
    git -C /app/FoundationStereo fetch --depth 1 origin ${FOUNDATIONSTEREO_COMMIT} && \
    git -C /app/FoundationStereo checkout FETCH_HEAD && \
    rm -rf /app/FoundationStereo/.git

# Pre-fetch the DINOv2 backbone (code + weights) into the torch hub cache so
# the GPU workers don't all race to download/extract it at once at runtime,
# which corrupts the cache (ENOTEMPTY errors).
ENV TORCH_HOME=/opt/torch HOME=/tmp
RUN python3 -c "import torch; torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14', pretrained=True)" && \
    chmod -R a+rwX /opt/torch

# Weights are baked in (not mounted).
WORKDIR /app
COPY generate_depth.py /app/generate_depth.py
COPY pretrained_models/23-51-11 /app/pretrained_models/23-51-11

ENTRYPOINT ["python3", "-u", "generate_depth.py"]
CMD ["--ckpt", "/app/pretrained_models/23-51-11/model_best_bp2.pth", "--missions", "/data/ETH-1"]
