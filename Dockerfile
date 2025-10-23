FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

# Set environment variables to prevent interactive prompts during installation.
ENV DEBIAN_FRONTEND=noninteractive

RUN sed -i 's/archive.ubuntu.com/mirrors.ocf.berkeley.edu/g' /etc/apt/sources.list

# Install Python
RUN apt-get update && \
    apt-get install -y python3-pip python3-dev python3-venv python-is-python3 wget git ninja-build vim libxcb1-dev libx11-dev libgl-dev pkgconf && \
    rm -rf /var/lib/apt/lists/* && \
    python -m pip install --upgrade pip

# Install CMake
RUN mkdir ~/temp && \
    cd ~/temp && \
    wget -nv https://github.com/Kitware/CMake/releases/download/v4.1.2/cmake-4.1.2-linux-x86_64.sh && \
    mkdir /opt/cmake && \
    sh cmake-4.1.2-linux-x86_64.sh --prefix=/usr/local --skip-license && \
    cmake --version

