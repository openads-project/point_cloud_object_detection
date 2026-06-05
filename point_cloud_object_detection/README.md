# `point_cloud_object_detection`

Detects objects in point clouds

The package provides a C++ ROS 2 component node for point-cloud object detection. It performs point-cloud preprocessing, calls a Triton-served model through `triton_cpp`, postprocesses bounding boxes, and publishes detected objects and optional visualization/debug outputs.

The exported `model_manifest.yml` is the source of truth for the model bundle. Its `frozen_contract` section defines the non-overridable model contract used by inference, and its `runtime_defaults` section provides default values for intentionally tunable runtime behavior.

Set `prediction.model_repository` to the exported Triton repository bundle root. The repository directory name identifies the exported artifact, while the Triton serving name is read from `config.pbtxt` and validated against `artifact.triton.model_name` in `model_manifest.yml`. `prediction.model_version` optionally selects the numbered Triton version directory. If it is empty, the export default from `model_manifest.yml` is used.

The node accepts `sensor_msgs/msg/PointCloud2` messages that always contain XYZ coordinates. The `input.point_feature_field` parameter controls whether the single feature channel is read from `intensity` or `reflectivity`. Additional feature channels beyond this single-feature setup are not supported.

Multiple instances of this node can run with separate namespaces and remapped topics. Shared memory transport is supported for multi-instance deployments: `triton_cpp` uses per-client shared-memory region names and only unregisters regions owned by that client instance, so one detection node does not clear another node's Triton registrations.

## Nodes

### `point_cloud_object_detection`

ROS 2 node for point-cloud object detection. The node supports private topic names, launch-time remapping, and multiple instances with separate namespaces.

**Subscribed Topics**

| Topic | Type | Description |
| --- | --- | --- |
| `~/point_cloud` | `sensor_msgs/msg/PointCloud2` | Input point cloud. |

**Published Topics**

| Topic | Type | Description |
| --- | --- | --- |
| `~/object_list` | `perception_msgs/msg/ObjectList` | Object list of detected objects. |
| `~/no_detection_zone` | `geometry_msgs/msg/PolygonStamped` | Polygon of the configured no-detection rectangle in `preprocessing.inference_frame`. |
| `~/no_detection_zone_points` | `sensor_msgs/msg/PointCloud2` | Raw points inside the no-detection zone. |
| `~/detection_area` | `geometry_msgs/msg/PolygonStamped` | Polygon of the configured detection sector in `preprocessing.inference_frame`. |
| `~/model_bounds` | `geometry_msgs/msg/PolygonStamped` | Polygon of the model XY bounds. |
| `~/density_grid_map` | `nav_msgs/msg/OccupancyGrid` | Density grid map decoded from the detection model output. |
| `~/occupancy_grid_map` | `nav_msgs/msg/OccupancyGrid` | Occupancy grid map decoded from the detection model output. |
| `~/combined_grid_map` | `nav_msgs/msg/OccupancyGrid` | Grid map that combines occupancy and density. |
| `~/static_grid_map` | `nav_msgs/msg/OccupancyGrid` | Static-structure grid map derived from density and occupancy outputs. |

All output topics are node-relative and stay in the node namespace unless remapped. Auxiliary grid maps are published in `output.grid_maps.frame` when set; otherwise they use `preprocessing.inference_frame`.

**Key Parameters**

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `prediction.server_url` | `string` | `127.0.0.1:8001` | Triton server host and gRPC port. |
| `prediction.model_repository` | `string` | `point-cloud-object-detection` | Path to the exported Triton model repository bundle root. |
| `prediction.model_version` | `string` | `""` | Requested Triton model version directory. Empty uses the export default from `model_manifest.yml`. |
| `prediction.triton_client_timeout_s` | `float` | `0.0` | Client timeout for Triton requests in seconds. `0.0` disables timeout. |
| `prediction.use_shm` | `bool` | `true` | Enable Triton shared-memory transport. |
| `prediction.cuda_input_shm` | `bool` | `false` | Require Triton CUDA shared memory for input tensors when CUDA preprocessing is used. |
| `point_cloud_transport` | `string` | `raw` | Transport hint used by the point-cloud subscriber. |
| `input.point_feature_field` | `string` | `intensity` | Point field used as the single model feature channel. |
| `preprocessing.backend` | `string` | `cpu` | Point preprocessing backend. Supported values are `cpu` and `cuda`. |
| `preprocessing.inference_frame` | `string` | `base_link` | Frame used for preprocessing and geometric filtering. |
| `preprocessing.no_detection_zone.enabled` | `bool` | `false` | Enable rectangular exclusion in `preprocessing.inference_frame`. |
| `preprocessing.detection_area.enabled` | `bool` | `false` | Enable circular-sector filtering and publishing. |
| `postprocessing.class_score_threshold` | `float` | model default | Minimum class score kept in the output list. |
| `postprocessing.nms.score_threshold` | `float[]` | model default | Candidate filtering and NMS score threshold. |
| `output.frame` | `string` | `base_link` | Frame reported in the output object list. |
| `output.sensor_id` | `int` | `0` | Sensor identifier stored on every object. |
| `output.variances` | `float[]` | `[-1.0, ...]` | Continuous-state covariance diagonal. Exactly 12 entries. |
| `output.grid_maps.frame` | `string` | `""` | Frame used for auxiliary grid-map publication. Empty uses `preprocessing.inference_frame`. |

At startup, invalid parameter values fail initialization. At runtime, invalid dynamic updates are rejected and the previous configuration is kept.

**Launch Example**

```bash
ros2 launch point_cloud_object_detection point_cloud_object_detection.launch.py \
  namespace:=/perception \
  params:=/docker-ros/ws/src/target/point_cloud_object_detection/config/params.yml \
  point_cloud_topic:=/my_lidar/points \
  object_list_topic:=/my_lidar/objects
```

**Launch Arguments**

| Argument | Default | Description |
| --- | --- | --- |
| `name` | `point_cloud_object_detection` | Node name. |
| `namespace` | `""` | Node namespace. |
| `params` | package `config/params.yml` | Path to the parameter file. |
| `log_level` | `info` | ROS logging level. |
| `use_sim_time` | `false` | Use simulation clock. |
| `trace` | `false` | Enable ROS tracing. |
| `point_cloud_topic` | `~/point_cloud` | Input point cloud topic remap. |
| `object_list_topic` | `~/object_list` | Output object list topic remap. |
| `no_detection_zone_topic` | `~/no_detection_zone` | No-detection-zone polygon topic remap. |
| `no_detection_zone_points_topic` | `~/no_detection_zone_points` | No-detection-zone points topic remap. |
| `detection_area_topic` | `~/detection_area` | Detection-area polygon topic remap. |
| `model_bounds_topic` | `~/model_bounds` | Model-bounds polygon topic remap. |
| `density_grid_map_topic` | `~/density_grid_map` | Density grid-map topic remap. |
| `occupancy_grid_map_topic` | `~/occupancy_grid_map` | Occupancy grid-map topic remap. |
| `combined_grid_map_topic` | `~/combined_grid_map` | Combined grid-map topic remap. |
| `static_grid_map_topic` | `~/static_grid_map` | Static grid-map topic remap. |

## Launch Files

### [`point_cloud_object_detection.launch.py`](launch/point_cloud_object_detection.launch.py)
