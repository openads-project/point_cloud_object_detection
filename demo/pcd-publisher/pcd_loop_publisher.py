#!/usr/bin/env python3

import os
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import rclpy
from builtin_interfaces.msg import Time
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header


def _parse_ascii_pcd(path: Path) -> List[Tuple[float, float, float, float]]:
    fields: List[str] = []
    data_started = False
    points: List[Tuple[float, float, float, float]] = []

    with path.open("r", encoding="ascii") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            upper = line.upper()
            if not data_started:
                if upper.startswith("FIELDS "):
                    fields = line.split()[1:]
                elif upper == "DATA ASCII":
                    if not fields:
                        raise ValueError("PCD file declares DATA before FIELDS")
                    data_started = True
                continue

            values = line.split()
            value_map = {name: values[idx] for idx, name in enumerate(fields)}
            points.append(
                (
                    float(value_map["x"]),
                    float(value_map["y"]),
                    float(value_map["z"]),
                    float(value_map["intensity"]),
                )
            )

    if not data_started:
        raise ValueError("Only ASCII PCD files are supported")
    if not points:
        raise ValueError(f"No points found in {path}")
    return points


def _make_cloud(points: Iterable[Tuple[float, float, float, float]], frame_id: str, stamp: Time) -> PointCloud2:
    fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1),
    ]
    header = Header(frame_id=frame_id, stamp=stamp)
    return point_cloud2.create_cloud(header, fields, points)


class PcdLoopPublisher(Node):
    def __init__(self) -> None:
        super().__init__("pcd_loop_publisher")

        pcd_dir = Path(os.environ.get("PCD_DIR", "/data"))
        pcd_glob = os.environ.get("PCD_GLOB", "fused_cloud_*.pcd")
        topic = os.environ.get("PCD_TOPIC", "/demo/points")
        frame_id = os.environ.get("PCD_FRAME_ID", "base_link")
        publish_rate_hz = float(os.environ.get("PCD_PUBLISH_RATE_HZ", "10.0"))

        self._pcd_paths = self._discover_pcd_files(pcd_dir, pcd_glob)
        self._frames = [_parse_ascii_pcd(path) for path in self._pcd_paths]
        self._frame_index = 0
        self._frame_id = frame_id

        qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self._publisher = self.create_publisher(PointCloud2, topic, qos)
        self._timer = self.create_timer(1.0 / publish_rate_hz, self._publish_once)

        self.get_logger().info(
            f"Loaded {len(self._frames)} PCD frames from {pcd_dir} using pattern {pcd_glob} and publishing on {topic} at {publish_rate_hz:.2f} Hz"
        )

    @staticmethod
    def _discover_pcd_files(pcd_dir: Path, pcd_glob: str) -> Sequence[Path]:
        paths = sorted(pcd_dir.glob(pcd_glob))
        if not paths:
            raise ValueError(f"No PCD files found in {pcd_dir} matching {pcd_glob}")
        return paths

    def _publish_once(self) -> None:
        points = self._frames[self._frame_index]
        msg = _make_cloud(points, self._frame_id, self.get_clock().now().to_msg())
        self._publisher.publish(msg)
        self._frame_index = (self._frame_index + 1) % len(self._frames)


def main() -> None:
    rclpy.init()
    node = PcdLoopPublisher()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
