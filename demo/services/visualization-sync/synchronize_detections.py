#!/usr/bin/env python3

# Copyright Institute for Automotive Engineering (ika), RWTH Aachen University
# SPDX-License-Identifier: Apache-2.0

"""Republish timestamp-matched point clouds and object lists for visualization."""

from __future__ import annotations

from collections import OrderedDict

import rclpy
from nav_msgs.msg import OccupancyGrid
from perception_msgs.msg import ObjectList
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSProfile, QoSReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import PointCloud2
from tf2_ros import Buffer, TransformException, TransformListener
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud


def _stamp_key(message) -> tuple[int, int]:
    """Return a hashable ROS timestamp key."""
    return message.header.stamp.sec, message.header.stamp.nanosec


class DetectionVisualizationSynchronizer(Node):
    """Match detector outputs to their source clouds by exact timestamp."""

    _GRID_MAP_TOPICS = (
        "density_grid_map",
        "dynamic_grid_map",
        "combined_grid_map",
        "static_grid_map",
    )

    def __init__(self) -> None:
        """Create synchronized publishers and input subscriptions."""
        super().__init__("detection_visualization_synchronizer")

        self.declare_parameter("cache_size", 32)
        self._cache_size = self.get_parameter("cache_size").get_parameter_value().integer_value
        if self._cache_size < 1:
            raise ValueError("cache_size must be positive")

        self._clouds: OrderedDict[tuple[int, int], PointCloud2] = OrderedDict()
        self._auxiliary_clouds: OrderedDict[tuple[int, int], PointCloud2] = OrderedDict()
        self._ground_truth_objects: OrderedDict[tuple[int, int], ObjectList] = OrderedDict()
        self._grid_maps: dict[str, OrderedDict[tuple[int, int], OccupancyGrid]] = {
            topic: OrderedDict() for topic in self._GRID_MAP_TOPICS
        }
        self._matched_stamps: OrderedDict[tuple[int, int], None] = OrderedDict()
        self._last_cloud_stamp: tuple[int, int] | None = None
        self._matched_count = 0
        self._missed_count = 0
        self._grid_map_publish_count = 0
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self, spin_thread=False)

        visualization_output_qos = QoSProfile(depth=1, reliability=QoSReliabilityPolicy.BEST_EFFORT)
        reliable_input_qos = QoSProfile(depth=self._cache_size, reliability=QoSReliabilityPolicy.RELIABLE)
        self._cloud_publisher = self.create_publisher(PointCloud2, "synchronized_point_cloud", visualization_output_qos)
        self._auxiliary_cloud_publisher = self.create_publisher(
            PointCloud2, "synchronized_auxiliary_point_cloud", visualization_output_qos
        )
        self._objects_publisher = self.create_publisher(ObjectList, "synchronized_object_list", 1)
        self._ground_truth_objects_publisher = self.create_publisher(ObjectList, "synchronized_ground_truth_object_list", 1)
        self._grid_map_publishers = {
            topic: self.create_publisher(OccupancyGrid, f"synchronized_{topic}", visualization_output_qos)
            for topic in self._GRID_MAP_TOPICS
        }
        self._input_subscriptions = [
            self.create_subscription(PointCloud2, "point_cloud", self._on_cloud, qos_profile_sensor_data),
            self.create_subscription(PointCloud2, "auxiliary_point_cloud", self._on_auxiliary_cloud, qos_profile_sensor_data),
            self.create_subscription(ObjectList, "object_list", self._on_objects, 1),
            self.create_subscription(ObjectList, "ground_truth_object_list", self._on_ground_truth_objects, 1),
        ]
        for topic in self._GRID_MAP_TOPICS:
            self._input_subscriptions.append(
                self.create_subscription(
                    OccupancyGrid, topic, lambda message, topic=topic: self._on_grid_map(topic, message), reliable_input_qos
                )
            )

        self.get_logger().info(f"Caching up to {self._cache_size} point clouds for timestamp synchronization")

    def _on_cloud(self, message: PointCloud2) -> None:
        stamp = _stamp_key(message)
        if self._last_cloud_stamp is not None and stamp < self._last_cloud_stamp:
            self.get_logger().info("Point-cloud time moved backwards; clearing synchronization cache")
            self._clouds.clear()
            self._auxiliary_clouds.clear()
            self._ground_truth_objects.clear()
            for grid_map_cache in self._grid_maps.values():
                grid_map_cache.clear()
            self._matched_stamps.clear()

        self._last_cloud_stamp = stamp
        self._clouds[stamp] = message
        self._clouds.move_to_end(stamp)
        while len(self._clouds) > self._cache_size:
            self._clouds.popitem(last=False)

    def _on_auxiliary_cloud(self, message: PointCloud2) -> None:
        stamp = _stamp_key(message)
        self._auxiliary_clouds[stamp] = message
        self._auxiliary_clouds.move_to_end(stamp)
        while len(self._auxiliary_clouds) > self._cache_size:
            self._auxiliary_clouds.popitem(last=False)

    def _on_objects(self, message: ObjectList) -> None:
        stamp = _stamp_key(message)
        cloud = self._clouds.get(stamp)
        if cloud is None:
            self._missed_count += 1
            if self._missed_count == 1 or self._missed_count % 50 == 0:
                self.get_logger().warning(
                    f"No cached point cloud for object-list timestamp {stamp[0]}.{stamp[1]:09d} " f"({self._missed_count} misses)"
                )
            return

        visualization_cloud = self._cloud_in_object_frame(cloud, message.header.frame_id)
        if visualization_cloud is None:
            return

        auxiliary_cloud = self._auxiliary_clouds.get(stamp)
        self._matched_stamps[stamp] = None
        self._matched_stamps.move_to_end(stamp)
        while len(self._matched_stamps) > self._cache_size:
            self._matched_stamps.popitem(last=False)

        ground_truth_objects = self._ground_truth_objects.pop(stamp, None)
        self._cloud_publisher.publish(visualization_cloud)
        if auxiliary_cloud is not None:
            self._auxiliary_cloud_publisher.publish(auxiliary_cloud)
        for topic, grid_map_cache in self._grid_maps.items():
            grid_map = grid_map_cache.pop(stamp, None)
            if grid_map is not None:
                self._publish_grid_map(topic, grid_map)
        if ground_truth_objects is not None:
            self._ground_truth_objects_publisher.publish(ground_truth_objects)
        self._objects_publisher.publish(message)

        self._matched_count += 1
        if self._matched_count == 1:
            self.get_logger().info("Publishing timestamp-synchronized visualization messages")

        self._discard_through(self._clouds, stamp)
        self._discard_through(self._auxiliary_clouds, stamp)
        for grid_map_cache in self._grid_maps.values():
            self._discard_through(grid_map_cache, stamp, inclusive=False)

    @staticmethod
    def _discard_through(cache: OrderedDict, stamp: tuple[int, int], *, inclusive: bool = True) -> None:
        """Evict timestamp-ordered cache entries without copying cache keys."""
        while cache:
            oldest_stamp = next(iter(cache))
            if oldest_stamp < stamp or inclusive and oldest_stamp == stamp:
                cache.popitem(last=False)
            else:
                return

    def _cloud_in_object_frame(self, cloud: PointCloud2, object_frame: str) -> PointCloud2 | None:
        """Transform a matched cloud into the frame in which detections are published."""
        if not object_frame or cloud.header.frame_id == object_frame:
            return cloud
        try:
            transform = self._tf_buffer.lookup_transform(
                object_frame,
                cloud.header.frame_id,
                Time.from_msg(cloud.header.stamp),
            )
        except TransformException as error:
            self.get_logger().warning(
                f"Cannot transform synchronized point cloud from {cloud.header.frame_id} "
                f"to {object_frame}: {error}"
            )
            return None
        return do_transform_cloud(cloud, transform)

    def _on_ground_truth_objects(self, message: ObjectList) -> None:
        stamp = _stamp_key(message)
        if stamp in self._matched_stamps:
            self._ground_truth_objects_publisher.publish(message)
            return

        self._ground_truth_objects[stamp] = message
        self._ground_truth_objects.move_to_end(stamp)
        while len(self._ground_truth_objects) > self._cache_size:
            self._ground_truth_objects.popitem(last=False)

    def _on_grid_map(self, topic: str, message: OccupancyGrid) -> None:
        stamp = _stamp_key(message)
        grid_map_cache = self._grid_maps[topic]
        grid_map_cache[stamp] = message
        grid_map_cache.move_to_end(stamp)
        while len(grid_map_cache) > self._cache_size:
            grid_map_cache.popitem(last=False)

        if stamp in self._matched_stamps:
            self._publish_grid_map(topic, grid_map_cache.pop(stamp))

    def _publish_grid_map(self, topic: str, message: OccupancyGrid) -> None:
        self._grid_map_publishers[topic].publish(message)
        self._grid_map_publish_count += 1
        if self._grid_map_publish_count == 1:
            self.get_logger().info("Publishing timestamp-synchronized grid maps")


def main() -> None:
    """Run the visualization synchronizer."""
    rclpy.init()
    node = DetectionVisualizationSynchronizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
