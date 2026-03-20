FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git ffmpeg libsm6 libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . .

# Install uv and project dependencies
RUN pip install uv && \
    uv sync

# Set the entrypoint to run the main command
ENTRYPOINT ["uv", "run", "ma-lucamertens"]