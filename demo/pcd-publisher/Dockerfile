FROM rwthika/ros2:jazzy-ros-base-v26.02

WORKDIR /opt/demo

COPY pcd_loop_publisher.py /opt/demo/pcd_loop_publisher.py

ENTRYPOINT ["/bin/bash", "-lc", "source /opt/ros/jazzy/setup.bash && python3 /opt/demo/pcd_loop_publisher.py"]
