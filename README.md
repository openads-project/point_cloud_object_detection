# Point Cloud Object Detection

This repository contains a C++ inference node for `Point Cloud Object Detection`. The ROS 2 node subscribes to a point cloud and publishes an object list.

This node does not perform the inference itself, but needs a [Triton server](https://gitlab.ika.rwth-aachen.de/fb-fi/ml/triton-server) to perform this task.

[[_TOC_]]

## Demo

A compact demo is provided in [demo/README.md](demo/README.md). It uses Docker Compose to start a Triton server, the packaged detection node, a point cloud publisher, RViz, and an `rqt` parameter GUI to interact with the node.

## Nodes

| Package | Node | Description |
| --- | --- | --- |
| `point_cloud_object_detection` | `point_cloud_object_detection` | ROS 2 node for `Point Cloud Object Detection` |

#### Subscribed Topics

| Topic | Type | Description |
| --- | --- | --- |
| `~/point_cloud` | `sensor_msgs/msg/PointCloud2` | Input point cloud |

#### Published Topics

| Topic | Type | Description |
| --- | --- | --- |
| `~/object_list` | `perception_msgs/msg/ObjectList` | Object list of detected objects |
| `~/no_detection_zone` | `geometry_msgs/msg/PolygonStamped` | Polygon of the configured no-detection rectangle (preprocessing.inference_frame) |
| `~/no_detection_zone_points` | `sensor_msgs/msg/PointCloud2` | Raw points inside the no-detection zone |
| `~/detection_area` | `geometry_msgs/msg/PolygonStamped` | Polygon of the configured detection sector (preprocessing.inference_frame) |
| `~/model_bounds` | `geometry_msgs/msg/PolygonStamped` | Polygon of the model xy bounds |
| `~/density_grid_map` | `nav_msgs/msg/OccupancyGrid` | Density grid map provided by the detection model; higher values correspond to a higher density of above-ground points |
| `~/occupancy_grid_map` | `nav_msgs/msg/OccupancyGrid` | Occupancy grid map provided by the detection model; higher values correspond to a higher probability of occupancy by detectable objects |
| `~/combined_grid_map` | `nav_msgs/msg/OccupancyGrid` | Grid map that combines occupancy and density into a single grid map |
| `~/static_grid_map` | `nav_msgs/msg/OccupancyGrid` | Grid map derived from the density grid map, with density attenuated where occupancy is high, to highlight static structures. |

All output topics are node-relative (start with `~`) and are always in the node's namespace.
Auxiliary grid maps are published in `output.grid_maps.frame` when set; otherwise they use `preprocessing.inference_frame`.

## Multi-Instance Support
- You can run multiple instances of this node, each with its own namespace and remapped topics.
- All output topics can be uniquely named per node instance.
- Shared memory transport is supported for multi-instance deployments. `triton_cpp` uses per-client shared-memory region names and only unregisters the regions owned by that client instance, so one detection node does not clear another node's Triton registrations.
- This applies both when multiple detection nodes share one Triton server and when each detection node talks to its own Triton server, as long as the underlying Triton shared-memory requirements are met.

## Launch File Usage
The provided [launch file](point_cloud_object_detection/launch/point_cloud_object_detection.launch.py) supports launch arguments and topic remappings.

Example usage:

```bash
ros2 launch point_cloud_object_detection point_cloud_object_detection.launch.py \
    namespace:=/perception \
    params:=/docker-ros/ws/src/target/point_cloud_object_detection/config/params.yml \
    point_cloud_topic:=/my_lidar/points \
    object_list_topic:=/my_lidar/objects
```

Set `prediction.model_repository` in the parameter file to the exported Triton repository bundle you want to use. The Triton serving name is read from `config.pbtxt` inside that repository.

### Launch Arguments ###

| Argument | Type | Description |
| --- | --- | --- |
| `name` | `string` | Node name (default: `point_cloud_object_detection`). |
| `namespace` | `string` | Node namespace (default: empty). |
| `params` | `string` | Path to parameter file (default: `point_cloud_object_detection/config/params.yml` from package share). |
| `log_level` | `string` | ROS log level (`debug|info|warn|error|fatal`, default: `info`). |
| `use_sim_time` | `bool` | Use simulation clock (`true|false`, default: `false`). |
| `trace` | `bool` | Enable tracing (`true|false`, default: `false`). |
| `point_cloud_topic` | `string` | Remap for input point cloud topic (default: `~/point_cloud`). |
| `object_list_topic` | `string` | Remap for output object list topic (default: `~/object_list`). |
| `no_detection_zone_topic` | `string` | Remap for no-detection zone polygon topic (default: `~/no_detection_zone`). |
| `no_detection_zone_points_topic` | `string` | Remap for no-detection-zone points topic (default: `~/no_detection_zone_points`). |
| `detection_area_topic` | `string` | Remap for detection area polygon topic (default: `~/detection_area`). |
| `model_bounds_topic` | `string` | Remap for model bounds polygon topic (default: `~/model_bounds`). |
| `density_grid_map_topic` | `string` | Remap for density grid map topic (default: `~/density_grid_map`). |
| `occupancy_grid_map_topic` | `string` | Remap for occupancy grid map topic (default: `~/occupancy_grid_map`). |
| `combined_grid_map_topic` | `string` | Remap for combined grid map topic (default: `~/combined_grid_map`). |
| `static_grid_map_topic` | `string` | Remap for static grid map topic (default: `~/static_grid_map`). |

## Parameters

At startup, invalid parameter values fail initialization. At runtime, invalid dynamic updates are rejected and the previous configuration is kept. The tables below list the allowed ranges and constraints.

**Prediction Parameters**

| Parameter | Type | Description | Constraints |
| --- | --- | --- | --- |
| `prediction.server_url` | `string` | Triton server host:port combination. | Required at startup. Read-only at runtime. |
| `prediction.model_repository` | `string` | Path to the exported Triton model repository bundle root. | Required at startup. Must point to a directory containing `model_manifest.yml` and `config.pbtxt`. Can be changed at runtime to reinitialize the model. |
| `prediction.model_version` | `string` | Requested Triton model version directory inside `prediction.model_repository`. | Optional at startup. If empty at startup, the export default from `model_manifest.yml` is used. Version-only runtime updates must name an existing version directory. |
| `prediction.triton_client_timeout_s` | `double` | Client timeout for Triton requests in seconds (`0.0` disables timeout). | Must be in `[0.0, 300.0]`. |
| `prediction.use_shm` | `bool` | Enable Triton shared-memory transport. | Requires client and Triton on the same host with a shared IPC namespace (e.g., Docker `ipc: host` or equivalent). |
| `prediction.cuda_input_shm` | `bool` | Require Triton CUDA shared memory for input tensors. Only used when `preprocessing.backend='cuda'`. | If enabled together with `preprocessing.backend='cuda'`, the node fails fast when CUDA SHM is unavailable or the CUDA-SHM path cannot be used. If `preprocessing.backend!='cuda'`, this setting is ignored with a warning. |

**Transport Parameters**

| Parameter | Type | Description | Constraints |
| --- | --- | --- | --- |
| `point_cloud_transport` | `string` | Transport hint used by the `point_cloud_transport` subscriber. | Must match an available point-cloud transport plugin. |

**Input Parameters**

| Parameter | Type | Description | Constraints |
| --- | --- | --- | --- |
| `input.point_feature_field` | `string` | Source for the single feature channel (`intensity` or `reflectivity`). | Must be `intensity` or `reflectivity`. |

**Preprocessing Parameters**

| Parameter | Type | Description | Constraints |
| --- | --- | --- | --- |
| `preprocessing.backend` | `string` | Point preprocessing backend (`cpu` or `cuda`). | If set to `cuda`, the node fails fast when CUDA preprocessing support is unavailable or the CUDA path cannot be used. |
| `preprocessing.inference_frame` | `string` | Frame used for preprocessing and geometric filtering. | Required. |
| `preprocessing.no_detection_zone.enabled` | `bool` | Enable rectangular exclusion in `preprocessing.inference_frame`. | - |
| `preprocessing.no_detection_zone.remove_points` | `bool` | Drop raw points that fall into the rectangle. | - |
| `preprocessing.no_detection_zone.publish_polygon` | `bool` | Publish the rectangle as `geometry_msgs/msg/PolygonStamped` on `~/no_detection_zone`. | - |
| `preprocessing.no_detection_zone.publish_points` | `bool` | Publish raw points inside the rectangle on `~/no_detection_zone_points`. | - |
| `preprocessing.no_detection_zone.x_min` | `double` | Lower x-bound of the rectangle. | Must be finite and satisfy `x_min < x_max` whenever the feature is enabled. |
| `preprocessing.no_detection_zone.x_max` | `double` | Upper x-bound of the rectangle. | Must be finite and satisfy `x_min < x_max`. |
| `preprocessing.no_detection_zone.y_min` | `double` | Lower y-bound of the rectangle. | Must be finite and satisfy `y_min < y_max`. |
| `preprocessing.no_detection_zone.y_max` | `double` | Upper y-bound of the rectangle. | Must be finite and satisfy `y_min < y_max`. |
| `preprocessing.detection_area.enabled` | `bool` | Enable circular-sector filtering/publishing. | - |
| `preprocessing.detection_area.center_x` | `double` | Sector centre x-position (metres). | Must be finite. |
| `preprocessing.detection_area.center_y` | `double` | Sector centre y-position (metres). | Must be finite. |
| `preprocessing.detection_area.radius` | `double` | Sector radius (metres). | Must be ≥ 0.0; must be > 0.0 when enabled. |
| `preprocessing.detection_area.bearing_deg` | `double` | Central azimuth in degrees (0 along +x, CCW positive). | Must lie within `[-360, 360]`. |
| `preprocessing.detection_area.fov_deg` | `double` | Sector aperture (degrees). | Must be within `(0, 360]`. |
| `preprocessing.detection_area.num_segments` | `int` | Number of segments used to approximate the arc. | Must be ≥ 3. |
| `preprocessing.detection_area.publish_polygon` | `bool` | Publish the sector as `geometry_msgs/msg/PolygonStamped` on `~/detection_area`. | - |
| `preprocessing.detection_area.filter_detections` | `bool` | Remove detections outside the configured detection area. Uses the XY sector together with `preprocessing.detection_area.z_min/z_max`. | - |
| `preprocessing.detection_area.filter_mode` | `string` | Filtering mode. `center` checks the box centroid in XY/Z; `complete` requires the full oriented XY footprint and full vertical extent to remain inside. | Must be either `center` or `complete`. |
| `preprocessing.detection_area.z_min` | `double` | Runtime override for the effective preprocessing lower z-bound used for point filtering and tensor construction. Defaults to the manifest z-range lower bound. | Must be finite and satisfy `z_min < z_max`. Values outside the manifest z range are accepted with a warning. |
| `preprocessing.detection_area.z_max` | `double` | Runtime override for the effective preprocessing upper z-bound used for point filtering and tensor construction. Defaults to the manifest z-range upper bound. | Must be finite and satisfy `z_min < z_max`. Values outside the manifest z range are accepted with a warning. |
| `preprocessing.point_feature.value_threshold` | `double` | Exported default for point-feature value-threshold normalization. | Defaults to `runtime_defaults.preprocessing.point_feature.value_threshold` from `model_manifest.yml`; must be finite and greater than 0 when value-threshold normalization is used. |

**Postprocessing Parameters**

| Parameter | Type | Description | Constraints |
| --- | --- | --- | --- |
| `postprocessing.class_score_threshold` | `double` | Minimum class score kept in the output list. | Defaults to `runtime_defaults.postprocessing.class_score_threshold` from `model_manifest.yml`; must be within `[0.0, 1.0]`. |
| `postprocessing.nms.score_threshold` | `double array` | Score threshold used for candidate filtering and NMS. | If unset, uses `runtime_defaults.postprocessing.nms.score_threshold` from `model_manifest.yml`; if set, must contain exactly one value or one per predicted class, and all entries must be within `[0.0, 1.0]`. |
| `postprocessing.nms.iou_threshold` | `double` | Optional NMS IoU threshold override. | If unset, uses `runtime_defaults.postprocessing.nms.iou_threshold` from `model_manifest.yml`; if set, must be within `[0.0, 1.0]`. |
| `postprocessing.nms.max_num_objects` | `int` | Optional maximum number of objects after NMS. | If unset, uses `runtime_defaults.postprocessing.nms.max_num_objects` from `model_manifest.yml`; if set, must be zero or positive. |

**Output Parameters**

| Parameter | Type | Description | Constraints |
| --- | --- | --- | --- |
| `output.frame` | `string` | Frame reported in the output object list. | Required. |
| `output.sensor_id` | `int` | Sensor identifier stored on every object. | Must be within `[0, 100000]`. |
| `output.variances` | `double array` | Continuous-state covariance diagonal. | Exactly 12 entries; each entry must be ≥ 0.0 or `-1.0` (`CONTINUOUS_STATE_COVARIANCE_UNKNOWN`). |
| `output.model_bounds.publish_polygon` | `bool` | Publish the xy bounds as `geometry_msgs/msg/PolygonStamped` on `~/model_bounds`. | - |
| `output.grid_maps.frame` | `string` | Frame used for auxiliary grid-map publication. If empty, the node publishes grid maps in `preprocessing.inference_frame`. | Optional. |
| `output.grid_maps.publish_density` | `bool` | Publish the decoded density auxiliary grid map on `~/density_grid_map`. This is useful for inspecting where the model sees strong local point support. | - |
| `output.grid_maps.publish_occupancy` | `bool` | Publish the decoded occupancy auxiliary grid map on `~/occupancy_grid_map`. This gives a rough view of which cells the model considers occupied. | - |
| `output.grid_maps.publish_combined` | `bool` | Publish the combined auxiliary grid map on `~/combined_grid_map`. It blends density and occupancy into a single map. | - |
| `output.grid_maps.publish_static` | `bool` | Publish the static auxiliary grid map on `~/static_grid_map`. This is intended to make more static scene structure stand out relative to dynamic occupancy cues. | - |
| `output.grid_maps.zero_in_no_detection_zone` | `bool` | If true, zero published auxiliary grid-map cells whose centers fall inside the configured no-detection rectangle. | Only applies when `preprocessing.no_detection_zone.enabled` is true. |
| `output.grid_maps.zero_outside_detection_area` | `bool` | If true, zero published auxiliary grid-map cells whose centers fall outside the configured detection area sector. | Only applies when `preprocessing.detection_area.enabled` is true. |
| `output.grid_maps.density_gain` | `double` | Linear gain applied to the published density grid map. | Must be finite and within `[0, 100]`. |
| `output.grid_maps.occupancy_gain` | `double` | Linear gain applied to the published occupancy grid map. | Must be finite and within `[0, 100]`. |
| `output.grid_maps.combined_gain` | `double` | Linear gain applied to the published combined grid map. | Must be finite and within `[0, 100]`. |
| `output.grid_maps.static_gain` | `double` | Linear gain applied to the published static grid map. | Must be finite and within `[0, 100]`. |

The exported `model_manifest.yml` is the source of truth for the bundle. Its `frozen_contract` section defines the exported model contract used by inference, and its `runtime_defaults` section provides the default values for intentionally tunable runtime behavior.
`params.yml` is the runtime selection and override file. `prediction.model_repository` selects the exported Triton repository bundle, `prediction.model_version` optionally selects the numbered Triton version directory, and the Triton model name is inferred from `config.pbtxt` inside that repository and validated against `artifact.triton.model_name` in `model_manifest.yml`.
`preprocessing.detection_area.z_min/z_max`, `preprocessing.point_feature.value_threshold`, `postprocessing.class_score_threshold`, `postprocessing.nms.score_threshold`, `postprocessing.nms.iou_threshold`, and `postprocessing.nms.max_num_objects` can override the exported defaults at runtime.
Changing `prediction.model_repository` at runtime reinitializes the model and preserves current runtime overrides, including `preprocessing.detection_area.z_min/z_max`.


## Usage of docker-ros Images

### Available Images

| Tag | Description |
| --- | --- |
| `latest` | latest ROS 2 version |

### Default Command

```bash
ros2 launch point_cloud_object_detection point_cloud_object_detection.launch.py
```

### Launch Files

| Package | File | Path | Description |
| --- | --- | --- | --- |
| `point_cloud_object_detection` | `point_cloud_object_detection.launch.py` | `/docker-ros/ws/install/share/point_cloud_object_detection/launch` | Launch the inference node using the configured parameter file |

### Configuration Files

| Package | File | Source Path | Installed Path | Description |
| --- | --- | --- | --- | --- |
| `point_cloud_object_detection` | `params.yml` | `/docker-ros/ws/src/target/point_cloud_object_detection/config/params.yml` | `/docker-ros/ws/install/point_cloud_object_detection/share/point_cloud_object_detection/config/params.yml` | Default runtime parameter file used by launch. |

## Point Cloud Input Fields

The node accepts `sensor_msgs/PointCloud2` messages that always contain XYZ coordinates. The `input.point_feature_field` parameter controls how the single feature channel is extracted before being forwarded to the model:

- `intensity` *(default)* – consumes the ROS `intensity` field as the single feature channel.
- `reflectivity` – consumes the ROS `reflectivity` field as the single feature channel. The field must exist and can be any numeric PointField datatype (`INT8/UINT8/INT16/UINT16/INT32/UINT32/FLOAT32/FLOAT64`); values are converted to `FLOAT32` internally.

Additional feature channels beyond this single-feature setup are not supported.
