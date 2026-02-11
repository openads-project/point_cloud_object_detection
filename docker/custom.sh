#!/bin/bash
set -e

# Detect architecture
ARCH=$(dpkg --print-architecture)

# Create a CMake cache file for architecture-specific settings
CMAKE_CACHE_DIR="/opt/ros-cmake-cache"
mkdir -p "$CMAKE_CACHE_DIR"

if [ "$ARCH" = "amd64" ]; then
    echo "Installing CUDA Toolkit 12.8 for amd64..."

    # Add NVIDIA CUDA repository
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
    dpkg -i cuda-keyring_1.1-1_all.deb
    rm cuda-keyring_1.1-1_all.deb

    # Update apt cache and install CUDA toolkit
    apt-get update
    apt-get install -y --no-install-recommends \
        cuda-toolkit-12-8

    # Clean up apt cache to reduce image size
    rm -rf /var/lib/apt/lists/*

    # Set CUDA environment variables for:
    # 1. Interactive bash shells
    echo 'export PATH=/usr/local/cuda-12.8/bin:$PATH' >> /etc/bash.bashrc
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH' >> /etc/bash.bashrc
    
    # 2. Non-interactive shells and Docker build stages
    echo 'PATH=/usr/local/cuda-12.8/bin:$PATH' >> /etc/environment
    
    # 3. Create symlink so CMake can find CUDA without full path
    ln -sf /usr/local/cuda-12.8 /usr/local/cuda

    # 4. Create marker file to indicate CUDA is available
    touch "$CMAKE_CACHE_DIR/cuda-available"

    echo "CUDA Toolkit 12.8 installation complete!"
else
    echo "Skipping CUDA installation on $ARCH (not supported)"
    echo "Building CPU-only version"
    
    # Create marker file to indicate CUDA is NOT available
    touch "$CMAKE_CACHE_DIR/cuda-unavailable"
fi
