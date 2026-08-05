FROM ros:jazzy-ros-core-noble AS runtime

ARG IMAGE_VERSION

LABEL org.opencontainers.image.source="https://github.com/openads-project/point_cloud_object_detection" \
      org.opencontainers.image.description="ROS 2 publisher and PCD files for the point cloud object detection demo" \
      org.opencontainers.image.version="${IMAGE_VERSION}"

RUN apt-get update && \
    apt-get install --no-install-recommends -y \
      ros-jazzy-sensor-msgs-py && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /opt/demo

COPY pcds/*.pcd /pcds/
COPY pcd_publisher.py /opt/demo/pcd_publisher.py

ENV PCD_DIR=/pcds

ENTRYPOINT ["/bin/bash", "-lc", "source /opt/ros/jazzy/setup.bash && python3 /opt/demo/pcd_publisher.py"]
