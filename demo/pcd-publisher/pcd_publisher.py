#!/usr/bin/env python3

# Copyright Institute for Automotive Engineering (ika), RWTH Aachen University
# SPDX-License-Identifier: Apache-2.0

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterable, List, Sequence, Tuple

import rclpy
from builtin_interfaces.msg import Time
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header


_PCD_TO_POINTFIELD_DATATYPES = {
    ("F", 4): PointField.FLOAT32,
    ("F", 8): PointField.FLOAT64,
    ("I", 1): PointField.INT8,
    ("I", 2): PointField.INT16,
    ("I", 4): PointField.INT32,
    ("U", 1): PointField.UINT8,
    ("U", 2): PointField.UINT16,
    ("U", 4): PointField.UINT32,
}

_TIMESTAMPED_PCD_PATTERN = re.compile(r"^(?P<index>\d+)_(?P<timestamp>\d+)$")
_NATURAL_SORT_TOKEN_PATTERN = re.compile(r"\d+|\D+")


@dataclass(frozen=True)
class PcdHeader:
    fields: List[str]
    sizes: List[int]
    types: List[str]
    counts: List[int]
    points: int
    data_mode: str

    @property
    def point_stride(self) -> int:
        return sum(size * count for size, count in zip(self.sizes, self.counts))

    def point_field_datatype_for_index(self, field_index: int) -> int:
        key = (self.types[field_index], self.sizes[field_index])
        point_field_datatype = _PCD_TO_POINTFIELD_DATATYPES.get(key)
        if point_field_datatype is None:
            raise ValueError(
                f"Unsupported PointCloud2 datatype mapping for '{self.fields[field_index]}': "
                f"type={self.types[field_index]} size={self.sizes[field_index]}"
            )
        return point_field_datatype


def _parse_pcd_header(handle: BinaryIO) -> PcdHeader:
    metadata: dict[str, List[str] | str] = {}

    while True:
        raw_line = handle.readline()
        if not raw_line:
            raise ValueError("PCD header ended before DATA declaration")

        line = raw_line.decode("ascii").strip()
        if not line or line.startswith("#"):
            continue

        parts = line.split()
        key = parts[0].upper()
        values = parts[1:]

        if key == "DATA":
            if not values:
                raise ValueError("PCD DATA line is missing the data mode")
            metadata["DATA"] = values[0].lower()
            break

        metadata[key] = values

    raw_fields = metadata.get("FIELDS")
    if not raw_fields:
        raise ValueError("PCD header is missing FIELDS")

    fields = [field.lower() for field in raw_fields]  # type: ignore[arg-type]
    sizes = [int(value) for value in metadata.get("SIZE", [])]  # type: ignore[arg-type]
    types = [value.upper() for value in metadata.get("TYPE", [])]  # type: ignore[arg-type]
    raw_counts = metadata.get("COUNT", [])  # type: ignore[assignment]
    counts = [int(value) for value in raw_counts] if raw_counts else [1] * len(fields)

    if not (len(fields) == len(sizes) == len(types) == len(counts)):
        raise ValueError("PCD header field metadata lengths do not match")

    raw_points = metadata.get("POINTS")
    if raw_points:
        points = int(raw_points[0])  # type: ignore[index]
    else:
        width = metadata.get("WIDTH")
        height = metadata.get("HEIGHT")
        if not width or not height:
            raise ValueError("PCD header is missing POINTS and WIDTH/HEIGHT")
        points = int(width[0]) * int(height[0])  # type: ignore[index]

    data_mode = metadata["DATA"]  # type: ignore[index]
    if data_mode not in {"ascii", "binary"}:
        raise ValueError(f"Unsupported PCD DATA mode '{data_mode}'")

    return PcdHeader(
        fields=fields,
        sizes=sizes,
        types=types,
        counts=counts,
        points=points,
        data_mode=data_mode,
    )


def _build_output_schema(header: PcdHeader) -> List[PointField]:
    for field_name in ("x", "y", "z"):
        if field_name not in header.fields:
            raise ValueError(f"PCD file is missing required field '{field_name}'")

    output_fields: List[PointField] = []
    offset = 0
    for index, field_name in enumerate(header.fields):
        output_fields.append(
            PointField(
                name=field_name,
                offset=offset,
                datatype=header.point_field_datatype_for_index(index),
                count=header.counts[index],
            )
        )
        offset += header.sizes[index] * header.counts[index]
    return output_fields


def _parse_ascii_points(
    handle: BinaryIO, header: PcdHeader, path: Path
) -> List[Tuple[object, ...]]:
    points: List[Tuple[object, ...]] = []

    for raw_line in handle:
        line = raw_line.decode("ascii").strip()
        if not line or line.startswith("#"):
            continue
        values = line.split()
        expected_value_count = sum(header.counts)
        if len(values) != expected_value_count:
            raise ValueError(
                f"PCD ASCII row width mismatch for {path}: expected {expected_value_count} values, got {len(values)}"
            )

        cursor = 0
        point: List[object] = []
        for field_type, count in zip(header.types, header.counts):
            parser = float if field_type == "F" else int
            for _ in range(count):
                point.append(parser(values[cursor]))
                cursor += 1
        points.append(tuple(point))

    if not points:
        raise ValueError(f"No points found in {path}")
    return points


def _load_binary_pcd_cloud(
    handle: BinaryIO, header: PcdHeader, fields: Sequence[PointField], frame_id: str, stamp: Time, path: Path
) -> PointCloud2:
    expected_size = header.points * header.point_stride
    point_data = handle.read(expected_size)
    if len(point_data) != expected_size:
        raise ValueError(
            f"PCD binary payload size mismatch for {path}: expected {expected_size} bytes, got {len(point_data)}"
        )

    msg = PointCloud2()
    msg.header = Header(frame_id=frame_id, stamp=stamp)
    msg.height = 1
    msg.width = header.points
    msg.fields = list(fields)
    msg.is_bigendian = False
    msg.point_step = header.point_stride
    msg.row_step = expected_size
    msg.data = point_data
    msg.is_dense = False
    return msg


def _load_pcd_cloud(path: Path, frame_id: str, stamp: Time) -> PointCloud2:
    with path.open("rb") as handle:
        header = _parse_pcd_header(handle)
        fields = _build_output_schema(header)

        if header.data_mode == "binary":
            return _load_binary_pcd_cloud(handle, header, fields, frame_id, stamp, path)
        if header.data_mode == "ascii":
            return _make_cloud(_parse_ascii_points(handle, header, path), fields, frame_id, stamp)
        raise ValueError(f"Unsupported PCD DATA mode '{header.data_mode}' in {path}")


def _natural_sort_key(name: str) -> Tuple[object, ...]:
    parts = _NATURAL_SORT_TOKEN_PATTERN.findall(name.lower())
    return tuple(int(part) if part.isdigit() else part for part in parts)


def _pcd_playback_sort_key(path: Path) -> Tuple[int, object, object]:
    stem_match = _TIMESTAMPED_PCD_PATTERN.match(path.stem)
    if stem_match is not None:
        timestamp = int(stem_match.group("timestamp"))
        index = int(stem_match.group("index"))
        return (0, timestamp, index)
    return (1, _natural_sort_key(path.stem), path.name.lower())


def _make_cloud(
    points: Iterable[Tuple[object, ...]], fields: Sequence[PointField], frame_id: str, stamp: Time
) -> PointCloud2:
    header = Header(frame_id=frame_id, stamp=stamp)
    return point_cloud2.create_cloud(header, list(fields), points)


class PcdPublisher(Node):
    def __init__(self) -> None:
        super().__init__("pcd_publisher")

        pcd_dir = Path(os.environ.get("PCD_DIR", "/data"))
        pcd_glob = os.environ.get("PCD_GLOB", "*.pcd")
        topic = os.environ.get("PCD_TOPIC", "/demo/points")
        frame_id = os.environ.get("PCD_FRAME_ID", "base_link")
        publish_rate_hz = float(os.environ.get("PCD_PUBLISH_RATE_HZ", "10.0"))

        pcd_paths = self._discover_pcd_files(pcd_dir, pcd_glob)
        self._frames = self._load_frames(pcd_paths, frame_id)
        self._frame_index = 0

        qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self._publisher = self.create_publisher(PointCloud2, topic, qos)
        self._timer = self.create_timer(1.0 / publish_rate_hz, self._publish_once)

        cached_bytes = sum(len(frame.data) for frame in self._frames)
        self.get_logger().info(
            f"Cached {len(self._frames)} PCD frames from {pcd_dir} using pattern {pcd_glob} "
            f"({cached_bytes / 1024 / 1024:.1f} MiB) and publishing on {topic} at {publish_rate_hz:.2f} Hz"
        )

    @staticmethod
    def _discover_pcd_files(pcd_dir: Path, pcd_glob: str) -> Sequence[Path]:
        paths = sorted(pcd_dir.glob(pcd_glob), key=_pcd_playback_sort_key)
        if not paths:
            raise ValueError(f"No PCD files found in {pcd_dir} matching {pcd_glob}")
        return paths

    @staticmethod
    def _load_frames(paths: Sequence[Path], frame_id: str) -> List[PointCloud2]:
        zero_stamp = Time(sec=0, nanosec=0)
        return [_load_pcd_cloud(path, frame_id, zero_stamp) for path in paths]

    def _publish_once(self) -> None:
        msg = self._frames[self._frame_index]
        msg.header.stamp = self.get_clock().now().to_msg()
        self._publisher.publish(msg)
        self._frame_index = (self._frame_index + 1) % len(self._frames)


def main() -> None:
    rclpy.init()
    node = PcdPublisher()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
