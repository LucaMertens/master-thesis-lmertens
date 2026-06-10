# Largely taken from https://github.com/raphaelschwinger/cuda-python-devcontainer/
ARG VARIANT=ubuntu22.04
ARG CUDA_VERSION=12.4.1
FROM nvidia/cuda:${CUDA_VERSION}-devel-${VARIANT}

# Set non-interactive mode for apt
ARG DEBIAN_FRONTEND=noninteractive

# Install base system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libcairo2-dev \
    pkg-config \
    python3-dev \
    python3-pip \
    git \
    curl \
    wget \
    bash-completion \
    ssh \
    vim \
    tmux \
    htop \
    nvtop \
    fish \
    sudo \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install uv system-wide
ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN UV_INSTALL_DIR=/usr/local/bin sh /uv-installer.sh && rm /uv-installer.sh

# Create a non-root user matching the host UID/GID
ARG USERNAME=lmertens
ARG USER_UID=1000
ARG USER_GID=1000

RUN set -eux; \
    if ! getent group "${USER_GID}" > /dev/null 2>&1; then \
        groupadd --gid "${USER_GID}" "${USERNAME}"; \
    fi; \
    if ! id -u "${USERNAME}" > /dev/null 2>&1; then \
        useradd --uid "${USER_UID}" --gid "${USER_GID}" -m --shell /usr/bin/fish "${USERNAME}"; \
    else \
        usermod --shell /usr/bin/fish "${USERNAME}"; \
    fi; \
    echo "${USERNAME} ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/${USERNAME}; \
    chmod 0440 /etc/sudoers.d/${USERNAME}

# Copy tmux config
COPY --chown=${USERNAME}:${USERNAME} config/.tmux.conf /home/${USERNAME}/.tmux.conf

# Copy fish tmux shortcuts
COPY --chown=${USERNAME}:${USERNAME} config/tmux.fish /home/${USERNAME}/.config/fish/conf.d/tmux.fish

# Install the Antigravity Jupyter Notebook MCP server system-wide via pip
RUN pip3 install --no-cache-dir \
    git+https://github.com/Ian747-tw/Antigravity-Jupiter-Notebook-MCP.git \
    ipykernel \
    jupyter_client

# Set DEBIAN_FRONTEND back to dialog
ARG DEBIAN_FRONTEND=dialog

USER ${USERNAME}

# Ensure uv and local pip bins are in PATH for the user
ENV PATH="/home/${USERNAME}/.local/bin:${PATH}"
