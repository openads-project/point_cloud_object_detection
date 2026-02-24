# Point Cloud Object Detection

This repository contains a C++ inference node for `Point Cloud Object Detection`. The ROS 2 node subscribes to a point cloud and publishes an object list.

This node does not perform the inference itself, but needs a [Triton server](https://gitlab.ika.rwth-aachen.de/fb-fi/ml/triton-server) to perform this task.

[[_TOC_]]

## Quickstart

1. Start the PCOD devcontainer in VSCode
1. Build the code with `Ctrl+Shift+B`
1. Run the PCOD with
    ```bash
    cd /docker-ros/ws
    . ./install/setup.bash
    ros2 launch point_cloud_object_detection point_cloud_object_detection.launch.py
    ```
    This will per default start PBOD and connect to a triton server deployed on FTH's workstation for testing purposes. Update `model_manifest_path` in [params.yml](point_cloud_object_detection/config/params.yml) to match the exported model bundle.
1. Give the node sime input, e.g. by playing a rosbag
    ```bash
    cd /docker-ros/ws
    . ./install/setup.bash
    ros2 bag play src/target/point_cloud_object_detection/models/rosbag2_2025_02_18-10_17_00/ --remap /points2:=/point_cloud_object_detection/point_cloud
    ```


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
| `~/no_detection_zone` | `geometry_msgs/msg/PolygonStamped` | Polygon of the configured no-detection rectangle (inference_frame) |
| `~/no_detection_zone_points` | `sensor_msgs/msg/PointCloud2` | Raw points inside the no-detection zone |
| `~/detection_area` | `geometry_msgs/msg/PolygonStamped` | Polygon of the configured detection sector (inference_frame) |
| `~/model_bounds` | `geometry_msgs/msg/PolygonStamped` | Polygon of the model xy bounds |

All output topics are node-relative (start with `~`) and are always in the node's namespace.

## Multi-Instance Support
- You can run multiple instances of this node, each with its own namespace and remapped topics.
- All output topics can be uniquely named per node instance.
- You **must** deactivate shared memory (SHM) for multiple instances, i.e., set `use_shm: False` in the [parameter file](point_cloud_object_detection/config/params.yml). 

## Launch File Usage
The provided [launch file](point_cloud_object_detection/launch/point_cloud_object_detection.launch.py) declares remappable topics for all inputs and outputs.

If your detection network predicts additional or different classes than the ones listed in the launch file, you need to add remapping arguments for those class-specific point cloud topics. For each of your classes, add a `DeclareLaunchArgument` for the topic, for example:

```python
DeclareLaunchArgument('class_point_clouds_myclass',
                      default_value='~/class_point_clouds/myclass'),
```

Example usage:

```bash
ros2 launch point_cloud_object_detection point_cloud_object_detection.launch.py \
    point_cloud_topic:=/my_lidar/points \
    object_list_topic:=/my_lidar/objects \
```

## Parameters

Invalid parameter values result in a fatal log message and the node shuts down. The tables below list the allowed ranges and constraints to keep the configuration valid.

**General Parameters**

| Parameter | Type | Description | Constraints |
| --- | --- | --- | --- |
| `point_cloud_transport` | `string` | Transport hint for the `point_cloud_transport` subscriber. | - |
| `model_name` | `string` | [**required**] Model name on the Triton server. | - |
| `model_version` | `string` | [**required**] Model version on the Triton server. Although numeric, it must be provided as a string. | - |
| `model_manifest_path` | `string` | Path (relative to the package) of the exported `model_manifest.yml`. | - |
| `point_feature_source` | `string` | [**dynamic**] Source for the single feature channel (`intensity` or `reflectivity`). | Must be `intensity` or `reflectivity`. |
| `server_url` | `string` | Triton server host:port combination. | - |
| `use_shm` | `bool` | Enable Triton shared-memory transport. | - |
| `inference_frame` | `string` | [**dynamic**] Frame used for preprocessing and filters. | - |
| `output_frame` | `string` | [**dynamic**] Frame reported in the output object list. | - |
| `sensor_id` | `int` | [**dynamic**] Sensor identifier stored on every object. | `-1` for a random id at start-up, otherwise a non-negative integer. |
| `variances` | `double array` | [**dynamic**] Continuous-state covariance diagonal. | Exactly 12 entries; each entry must be ≥ 0.0 or `-1.0` (`CONTINUOUS_STATE_COVARIANCE_UNKNOWN`). |
| `class_score_threshold` | `double` | [**dynamic**] Minimum class score kept in the output list. | Must be within `[0.0, 1.0]`. |
| `publish_class_point_clouds` | `bool` | Publish per-class point clouds. | - |
| `publish_unclassified_points` | `bool` | Publish points not covered by any bounding box. | - |

**No-Detection Zone (inference_frame)**

| Parameter | Type | Description | Constraints |
| --- | --- | --- | --- |
| `no_detection_zone.enabled` | `bool` | Enable rectangular exclusion in `inference_frame`. | - |
| `no_detection_zone.remove_points` | `bool` | Drop raw points that fall into the rectangle. | - |
| `no_detection_zone.publish_polygon` | `bool` | Publish the rectangle as `geometry_msgs/msg/PolygonStamped` on `~/no_detection_zone`. | - |
| `no_detection_zone.publish_points` | `bool` | Publish raw points inside the rectangle on `~/no_detection_zone_points`. | - |
| `no_detection_zone.x_min` | `double` | Lower x-bound of the rectangle. | Must be finite and satisfy `x_min < x_max` whenever the feature is enabled. |
| `no_detection_zone.x_max` | `double` | Upper x-bound of the rectangle. | Must be finite and satisfy `x_min < x_max`. |
| `no_detection_zone.y_min` | `double` | Lower y-bound of the rectangle. | Must be finite and satisfy `y_min < y_max`. |
| `no_detection_zone.y_max` | `double` | Upper y-bound of the rectangle. | Must be finite and satisfy `y_min < y_max`. |

If `remove_points` or `enabled` is true and the bounds are invalid, the node logs a fatal error and exits.

**Detection Area (inference_frame)**

| Parameter | Type | Description | Constraints |
| --- | --- | --- | --- |
| `detection_area.enabled` | `bool` | Enable circular-sector filtering/publishing. | - |
| `detection_area.center_x` | `double` | Sector centre x-position (metres). | Must be finite. |
| `detection_area.center_y` | `double` | Sector centre y-position (metres). | Must be finite. |
| `detection_area.radius` | `double` | Sector radius (metres). | Must be ≥ 0.0; must be > 0.0 when the detection area is enabled. |
| `detection_area.bearing_deg` | `double` | Central azimuth in degrees (0 along +x, CCW positive). | Must lie within `[-360, 360]`. |
| `detection_area.fov_deg` | `double` | Sector aperture (degrees). | Must be within `(0, 360]`. |
| `detection_area.num_segments` | `int` | Number of segments used to approximate the arc. | Must be ≥ 3. |
| `detection_area.publish_polygon` | `bool` | Publish the sector as `geometry_msgs/msg/PolygonStamped` on `~/detection_area`. | - |
| `detection_area.filter_detections` | `bool` | Remove detections outside the sector. | - |
| `detection_area.filter_mode` | `string` | Filtering mode. | Must be either `center` or `complete`. |
| `publish_unclassified_points_outside_detection_area` | `bool` | Publish unclassified points that lie outside the configured sector. | Only meaningful when the detection area is enabled with a positive radius. |

Any violation of the detection area constraints above causes the node to emit a fatal error and terminate.

**Model Bounds Parameters**

| Parameter | Type | Description | Constraints |
| --- | --- | --- | --- |
| `model_bounds.publish_polygon` | `bool` | Publish the xy bounds as `geometry_msgs/msg/PolygonStamped` on `~/model_bounds`. | - |

Model-specific parameters (grid size, class names, stride, etc.) are loaded from the exported `model_manifest.yml`.

**NMS parameters (overrides manifest)**

| Parameter | Type | Description | Constraints |
| --- | --- | --- | --- |
| `nms_max_num_objects` | `int` | [**dynamic**] Maximum number of objects considered during NMS. | Must be ≥ 0. |
| `nms_iou_threshold` | `double` | [**dynamic**] IoU threshold for suppression. | Must be within `[0.0, 1.0]`. |
| `nms_score_threshold` | `double array` | [**dynamic**] Minimum score(s) used during suppression. | Provide either one value or one per class; every entry must lie within `[0.0, 1.0]`. |


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

| Package | File | Path | Description |
| --- | --- | --- | --- |
| `point_cloud_object_detection` | `params.yml` | `/docker-ros/ws/install/point_cloud_object_detection/share/point_cloud_object_detection/config` | File containing static parameters |
| `point_cloud_object_detection` | `config.yml` | `/docker-ros/ws/install/point_cloud_object_detection/share/point_cloud_object_detection/config` | File containing model config parameters |


## Point Cloud Input Fields

The node accepts `sensor_msgs/PointCloud2` messages that always contain XYZ coordinates. The `point_feature_source` parameter controls how the single feature channel is extracted before being forwarded to the model:

- `intensity` *(default)* – consumes the ROS `intensity` field as the single feature channel.
- `reflectivity` – consumes the ROS `reflectivity` field as the single feature channel. The field must exist and can be any numeric PointField datatype (`INT8/UINT8/INT16/UINT16/INT32/UINT32/FLOAT32/FLOAT64`); values are converted to `FLOAT32` internally.

Additional feature channels beyond this single-feature setup are not supported.
