FROM rwthika/ros2:jazzy-ros-base

WORKDIR /opt/demo

COPY pcd_publisher.py /opt/demo/pcd_publisher.py

ENTRYPOINT ["/bin/bash", "-lc", "source /opt/ros/jazzy/setup.bash && python3 /opt/demo/pcd_publisher.py"]
