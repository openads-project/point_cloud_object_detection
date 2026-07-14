#!/usr/bin/env python3

# Copyright Institute for Automotive Engineering (ika), RWTH Aachen University
# SPDX-License-Identifier: Apache-2.0

"""Convert synchronized occupancy-grid layers into colored images."""

from __future__ import annotations

from collections import OrderedDict
from typing import Final

import numpy as np
import rclpy
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Image


Color = tuple[int, int, int]
PaletteStops = tuple[Color, ...]
UNKNOWN_COLOR: Final = (7, 18, 31)


def _stamp_key(message: OccupancyGrid) -> tuple[int, int]:
    """Return a hashable ROS timestamp key."""
    return message.header.stamp.sec, message.header.stamp.nanosec


def _color_lookup(stops: PaletteStops) -> np.ndarray:
    """Build a 0--100 RGB lookup table from evenly spaced color stops."""
    lookup = np.full((256, 3), UNKNOWN_COLOR, dtype=np.uint8)
    for value in range(101):
        position = value / 100.0 * (len(stops) - 1)
        index = min(int(position), len(stops) - 2)
        fraction = position - index
        lookup[value] = tuple(round(start + (end - start) * fraction) for start, end in zip(stops[index], stops[index + 1]))
    lookup[101:128] = lookup[100]
    return lookup


class GridMapImageRepublisher(Node):
    """Publish RGB grid-map images using the NVIDIA demo color scheme."""

    _LAYERS: Final = ("density", "dynamic", "combined", "static")
    _PALETTES: Final[dict[str, np.ndarray]] = {
        "density": _color_lookup(((7, 18, 31), (23, 79, 112), (103, 232, 210), (242, 248, 255))),
        "dynamic": _color_lookup(((7, 18, 31), (68, 51, 122), (222, 90, 138), (255, 209, 102))),
        "static": _color_lookup(((7, 18, 31), (36, 81, 75), (103, 232, 210), (247, 250, 252))),
    }

    def __init__(self) -> None:
        super().__init__("grid_map_image_republisher")

        self.declare_parameter("cache_size", 32)
        self._cache_size = self.get_parameter("cache_size").get_parameter_value().integer_value
        if self._cache_size < 1:
            raise ValueError("cache_size must be positive")
        self._combined_layers: dict[str, OrderedDict[tuple[int, int], OccupancyGrid]] = {
            layer: OrderedDict() for layer in ("density", "dynamic")
        }

        input_qos = QoSProfile(depth=1, reliability=QoSReliabilityPolicy.BEST_EFFORT)
        image_qos = QoSProfile(depth=1, reliability=QoSReliabilityPolicy.BEST_EFFORT)
        self._publishers = {
            layer: self.create_publisher(Image, f"{layer}_grid_map_image", image_qos) for layer in self._LAYERS
        }
        self._subscriptions = [
            self.create_subscription(
                OccupancyGrid,
                f"{layer}_grid_map",
                lambda message, layer=layer: self._on_grid_map(layer, message),
                input_qos,
            )
            for layer in ("density", "dynamic", "static")
        ]
        self.get_logger().info("Republishing synchronized grid maps as colored rgb8 images")

    def _on_grid_map(self, layer: str, grid_map: OccupancyGrid) -> None:
        if layer in self._combined_layers:
            self._cache_combined_layer(layer, grid_map)
        image = self._to_image(grid_map, self._PALETTES[layer])
        if image is not None:
            self._publishers[layer].publish(image)
        else:
            self._warn_malformed(layer, grid_map)

    def _cache_combined_layer(self, layer: str, grid_map: OccupancyGrid) -> None:
        stamp = _stamp_key(grid_map)
        cache = self._combined_layers[layer]
        cache[stamp] = grid_map
        cache.move_to_end(stamp)
        while len(cache) > self._cache_size:
            cache.popitem(last=False)

        other_layer = "dynamic" if layer == "density" else "density"
        other_grid_map = self._combined_layers[other_layer].pop(stamp, None)
        if other_grid_map is None:
            return
        cache.pop(stamp, None)
        density_grid_map = grid_map if layer == "density" else other_grid_map
        dynamic_grid_map = grid_map if layer == "dynamic" else other_grid_map
        image = self._combined_to_image(density_grid_map, dynamic_grid_map)
        if image is None:
            self.get_logger().warning("Ignoring incompatible density and dynamic grid maps for combined image")
            return
        self._publishers["combined"].publish(image)

    def _warn_malformed(self, layer: str, grid_map: OccupancyGrid) -> None:
        self.get_logger().warning(
            f"Ignoring malformed {layer} grid map: expected "
            f"{grid_map.info.width * grid_map.info.height} cells, received {len(grid_map.data)}"
        )

    @classmethod
    def _to_image(cls, grid_map: OccupancyGrid, palette: np.ndarray) -> Image | None:
        """Color a grid with a lookup table and mirror it left-to-right."""
        width = grid_map.info.width
        height = grid_map.info.height
        if width == 0 or height == 0 or len(grid_map.data) != width * height:
            return None

        cells = np.frombuffer(grid_map.data, dtype=np.uint8, count=width * height).reshape(height, width)
        return cls._image_from_pixels(grid_map, palette[cells[:, ::-1]].tobytes())

    @classmethod
    def _combined_to_image(cls, density_grid_map: OccupancyGrid, dynamic_grid_map: OccupancyGrid) -> Image | None:
        """Render the density base with the dynamic layer overlaid."""
        if (
            density_grid_map.info.width != dynamic_grid_map.info.width
            or density_grid_map.info.height != dynamic_grid_map.info.height
            or len(density_grid_map.data) != len(dynamic_grid_map.data)
        ):
            return None

        width = density_grid_map.info.width
        height = density_grid_map.info.height
        if width == 0 or height == 0 or len(density_grid_map.data) != width * height:
            return None
        density_cells = np.frombuffer(density_grid_map.data, dtype=np.uint8, count=width * height).reshape(
            height, width
        )[:, ::-1]
        dynamic_cells = np.frombuffer(dynamic_grid_map.data, dtype=np.uint8, count=width * height).reshape(
            height, width
        )[:, ::-1]
        pixels = cls._PALETTES["density"][density_cells]
        dynamic_mask = (dynamic_cells > 0) & (dynamic_cells < 128)
        pixels[dynamic_mask] = cls._PALETTES["dynamic"][dynamic_cells[dynamic_mask]]
        return cls._image_from_pixels(density_grid_map, pixels.tobytes())

    @staticmethod
    def _image_from_pixels(grid_map: OccupancyGrid, pixels: bytes) -> Image:
        """Create an rgb8 image that preserves the grid-map header and layout."""
        image = Image()
        image.header = grid_map.header
        image.height = grid_map.info.height
        image.width = grid_map.info.width
        image.encoding = "rgb8"
        image.is_bigendian = False
        image.step = image.width * 3
        image.data = pixels
        return image


def main() -> None:
    """Run the grid-map image republisher."""
    rclpy.init()
    node = GridMapImageRepublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
