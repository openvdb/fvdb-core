# Start from the official NVIDIA CUDA development image.
# This matches fvdb's requirement for CUDA 12.0 and provides a full Ubuntu 22.04 OS.
FROM nvidia/cuda:12.0.1-devel-ubuntu22.04

# Set environment variables to prevent interactive prompts during installation.
ENV DEBIAN_FRONTEND=noninteractive

# Install basic system dependencies and developer tools.
# sudo is critical for the devcontainer common-utils feature to work.
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    git \
    sudo \
    && rm -rf /var/lib/apt/lists/*

# Delete the non-root user created by the base image.
RUN if grep ubuntu:x:1000:1000 /etc/passwd >/dev/null; then userdel -f -r ubuntu; fi

# Download and install Miniforge (Conda).
# This installs Conda into a standard location.
RUN wget \
    "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh" \
    -O /tmp/miniforge.sh && \
    bash /tmp/miniforge.sh -b -p /opt/conda && \
    rm /tmp/miniforge.sh

# Add Conda to the system's PATH for all users and initialize it.
# This ensures that 'conda' is available in shells
ENV PATH="/opt/conda/bin:${PATH}"
#RUN conda init bash

# IMPORTANT: We do NOT add a USER, CMD, or ENTRYPOINT here.
# User creation will be handled by the 'common-utils' feature.
# The container lifecycle (CMD/ENTRYPOINT) will be handled by Docker Compose.
