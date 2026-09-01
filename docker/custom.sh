#!/bin/bash
set -e

ARCH=$(dpkg --print-architecture)

CMAKE_CACHE_DIR="/opt/ros-cmake-cache"
mkdir -p "$CMAKE_CACHE_DIR"

if [ "$ARCH" = "amd64" ]; then
    echo "Installing CUDA Toolkit 12.8 for amd64..."

    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
    dpkg -i cuda-keyring_1.1-1_all.deb
    rm cuda-keyring_1.1-1_all.deb

    apt-get update
    apt-get install -y --no-install-recommends cuda-toolkit-12-8

    echo 'export PATH=/usr/local/cuda-12.8/bin:$PATH' >> /etc/bash.bashrc
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH' >> /etc/bash.bashrc
    echo 'PATH=/usr/local/cuda-12.8/bin:$PATH' >> /etc/environment

    ln -sf /usr/local/cuda-12.8 /usr/local/cuda
    touch "$CMAKE_CACHE_DIR/cuda-available"

    echo "CUDA Toolkit 12.8 installation complete!"
else
    echo "Skipping CUDA installation on $ARCH (not supported)"
    echo "Building CPU-only version"
    touch "$CMAKE_CACHE_DIR/cuda-unavailable"
fi

# TODO: remove once [https://github.com/ros2/ros2_tracing/issues/211] is solved in released version
# overwrite released ros2_tracing packages with fork to support
# 'message-link instrumentation' and 'dual-session mode' in jazzy
cd /docker-ros/ws
git clone --branch jazzy-ika https://github.com/RaphvK/ros2_tracing.git src/ros2_tracing
rosdep update && rosdep install -y -i --from-paths src/ros2_tracing
source /opt/ros/${ROS_DISTRO}/setup.bash
colcon build --cmake-args '-DCMAKE_BUILD_TYPE=Release' --allow-overriding tracetools --allow-overriding tracetools_launch
rm -r src/ros2_tracing log build