# Point Cloud Object Detection

This repository contains a C++ inference node for `Point Cloud Object Detection`. The ROS 2 node subscribes to a point cloud and publishes an object list.

This node does not perform the inference itself, but needs a [Triton server](https://gitlab.ika.rwth-aachen.de/fb-fi/ml/triton-server) to perform this task.

[[_TOC_]]

## Demo

A compact demo is provided in [demo/README.md](/docker-ros/ws/src/target/demo/README.md). It uses Docker Compose to start a Triton server, the packaged detection node, a point cloud publisher, RViz, and an `rqt` parameter GUI to interact with the node.

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

All output topics are node-relative (start with `~`) and are always in the node's namespace.

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

Set `prediction.model_repository_path` in the parameter file to the exported Triton repository bundle you want to use. The repository directory name identifies the exported artifact, while the Triton serving name is read from `config.pbtxt` inside that repository.

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

## Parameters

At startup, invalid parameter values fail initialization. At runtime, invalid dynamic updates are rejected and the previous configuration is kept. The tables below list the allowed ranges and constraints.

**Prediction Parameters**

| Parameter | Type | Description | Constraints |
| --- | --- | --- | --- |
| `prediction.server_url` | `string` | Triton server host:port combination. | Required at startup. Read-only at runtime. |
| `prediction.model_repository_path` | `string` | Path to the exported Triton model repository bundle root. | Required at startup. Must point to a directory containing `model_manifest.yml` and `config.pbtxt`. Read-only at runtime. |
| `prediction.model_version` | `string` | Requested Triton model version directory inside `prediction.model_repository_path`. | Optional. If empty, the export default from `model_manifest.yml` is used. The resolved version directory must exist. Read-only at runtime. |
| `prediction.triton_client_timeout_s` | `double` | [**dynamic**] Client timeout for Triton requests in seconds (`0.0` disables timeout). | Must be in `[0.0, 300.0]`. |
| `prediction.use_shm` | `bool` | Enable Triton shared-memory transport. | Requires client and Triton on the same host with a shared IPC namespace (e.g., Docker `ipc: host` or equivalent). |
| `prediction.cuda_input_shm` | `bool` | Require Triton CUDA shared memory for input tensors. Only used when `preprocessing.backend='cuda'`. | If enabled together with `preprocessing.backend='cuda'`, the node fails fast when CUDA SHM is unavailable or the CUDA-SHM path cannot be used. If `preprocessing.backend!='cuda'`, this setting is ignored with a warning. |

**Transport Parameters**

| Parameter | Type | Description | Constraints |
| --- | --- | --- | --- |
| `point_cloud_transport` | `string` | [**dynamic**] Transport hint used by the `point_cloud_transport` subscriber. | Must match an available point-cloud transport plugin. |

**Input Parameters**

| Parameter | Type | Description | Constraints |
| --- | --- | --- | --- |
| `input.point_feature_field` | `string` | [**dynamic**] Source for the single feature channel (`intensity` or `reflectivity`). | Must be `intensity` or `reflectivity`. |

**Preprocessing Parameters**

| Parameter | Type | Description | Constraints |
| --- | --- | --- | --- |
| `preprocessing.backend` | `string` | [**dynamic**] Point preprocessing backend (`cpu` or `cuda`). | If set to `cuda`, the node fails fast when CUDA preprocessing support is unavailable or the CUDA path cannot be used. |
| `preprocessing.inference_frame` | `string` | [**dynamic**] Frame used for preprocessing and geometric filtering. | Required. |
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
| `preprocessing.detection_area.filter_detections` | `bool` | Remove detections outside the sector. | - |
| `preprocessing.detection_area.filter_mode` | `string` | Filtering mode. | Must be either `center` or `complete`. |
| `preprocessing.point_feature.value_threshold` | `double` | [**dynamic**] Exported default for point-feature value-threshold normalization. | Defaults to `runtime_defaults.preprocessing.point_feature.value_threshold` from `model_manifest.yml`; must be finite and greater than 0 when value-threshold normalization is used. |

**Postprocessing Parameters**

| Parameter | Type | Description | Constraints |
| --- | --- | --- | --- |
| `postprocessing.class_score_threshold` | `double` | [**dynamic**] Minimum class score kept in the output list. | Defaults to `runtime_defaults.postprocessing.class_score_threshold` from `model_manifest.yml`; must be within `[0.0, 1.0]`. |
| `postprocessing.nms.score_threshold` | `double array` | [**dynamic**] Score threshold used for candidate filtering and NMS. | If unset, uses `runtime_defaults.postprocessing.nms.score_threshold` from `model_manifest.yml`; if set, must contain exactly one value or one per predicted class, and all entries must be within `[0.0, 1.0]`. |
| `postprocessing.nms.iou_threshold` | `double` | [**dynamic**] Optional NMS IoU threshold override. | If unset, uses `runtime_defaults.postprocessing.nms.iou_threshold` from `model_manifest.yml`; if set, must be within `[0.0, 1.0]`. |
| `postprocessing.nms.max_num_objects` | `int` | [**dynamic**] Optional maximum number of objects after NMS. | If unset, uses `runtime_defaults.postprocessing.nms.max_num_objects` from `model_manifest.yml`; if set, must be zero or positive. |

**Output Parameters**

| Parameter | Type | Description | Constraints |
| --- | --- | --- | --- |
| `output.frame` | `string` | [**dynamic**] Frame reported in the output object list. | Required. |
| `output.sensor_id` | `int` | [**dynamic**] Sensor identifier stored on every object. | Must be within `[0, 100000]`. |
| `output.variances` | `double array` | [**dynamic**] Continuous-state covariance diagonal. | Exactly 12 entries; each entry must be ≥ 0.0 or `-1.0` (`CONTINUOUS_STATE_COVARIANCE_UNKNOWN`). |
| `output.model_bounds.publish_polygon` | `bool` | Publish the xy bounds as `geometry_msgs/msg/PolygonStamped` on `~/model_bounds`. | - |

The exported `model_manifest.yml` is the source of truth for the bundle. Its `frozen_contract` section defines the non-overridable model contract used by inference, and its `runtime_defaults` section provides the default values for intentionally tunable runtime behavior.
`params.yml` is the runtime selection and override file. `prediction.model_repository_path` selects the exported Triton repository bundle, `prediction.model_version` optionally selects the numbered Triton version directory, and the Triton model name is inferred from `config.pbtxt` inside that repository and validated against `artifact.triton.model_name` in `model_manifest.yml`. Repository directory names do not need to match the Triton model name.
`preprocessing.point_feature.value_threshold`, `postprocessing.class_score_threshold`, `postprocessing.nms.score_threshold`, `postprocessing.nms.iou_threshold`, and `postprocessing.nms.max_num_objects` can override the exported defaults at runtime.


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
