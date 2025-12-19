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
    This will per default start PBOD and connect to a triton server deployed on FTH's workstation for testing purposes. Also a PP model is running there, which can be tested by setting the `model_config` to `models/pp_triton.yml` in the [params.yml](point_cloud_object_detection/config/params.yml) file.
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
| `~/class_point_cloud/<class_name>` | `sensor_msgs/msg/PointCloud2` | Per-class point clouds (if enabled). Each class predicted by the network gets its own topic. |
| `~/class_point_cloud` | `sensor_msgs/msg/PointCloud2` | Points not assigned to any object (if enabled). |

All output topics are node-relative (start with `~`) and are always in the node's namespace.

## Multi-Instance Support
- You can run multiple instances of this node, each with its own namespace and remapped topics.
- All output topics can be uniquely named per node instance.
- You **must** deactivate shared memory (SHM) for multiple instances, i.e., set `use_shm: False` in the [parameter file](point_cloud_object_detection/config/params.yml). 

## Launch File Usage
The provided [launch files](point_cloud_object_detection/launch/point_cloud_object_detection.launch.py) declare remappable topics for all inputs and outputs. 

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
    class_point_cloud_car_topic:=/my_lidar/car \
    class_point_cloud_pedestrian_topic:=/my_lidar/pedestrian \
    class_point_cloud_bicycle_topic:=/my_lidar/truck \
    class_point_cloud_truck_topic:=/my_lidar/trailer \
    class_point_cloud_bus_topic:=/my_lidar/bus \
    class_point_cloud_two_wheeler_topic:=/my_lidar/two_wheeler \
    unclassified_point_cloud_topic:=/my_lidar/unclassified
```

## Parameters

Invalid parameter values result in a fatal log message and the node shuts down. The tables below list the allowed ranges and constraints to keep the configuration valid.

**General Parameters**

| Parameter | Type | Description | Constraints |
| --- | --- | --- | --- |
| `point_cloud_transport` | `string` | Transport hint for the `point_cloud_transport` subscriber. | - |
| `model_name` | `string` | [**required**] Model name on the Triton server. | - |
| `model_version` | `string` | [**required**] Model version on the Triton server. Although numeric, it must be provided as a string. | - |
| `model_config` | `string` | Path (relative to the package) of the model configuration YAML. | - |
| `server_url` | `string` | Triton server host:port combination. | - |
| `use_shm` | `bool` | Enable Triton shared-memory transport. | - |
| `inference_frame` | `string` | [**dynamic**] Frame used for preprocessing and filters. | - |
| `output_frame` | `string` | [**dynamic**] Frame reported in the output object list. | - |
| `sensor_id` | `int` | [**dynamic**] Sensor identifier stored on every object. | `-1` for a random id at start-up, otherwise a non-negative integer. |
| `variances` | `double array` | [**dynamic**] Continuous-state covariance diagonal. | Exactly 12 entries; each entry must be ≥ 0.0 or `-1.0` (`CONTINUOUS_STATE_COVARIANCE_UNKNOWN`). |
| `class_score_threshold` | `double` | [**dynamic**] Minimum class score kept in the output list. | Must be within `[0.0, 1.0]`. |
| `mask_is_bool` | `bool` | PBOD compatibility switch for mask dtype. | - |
| `zero_intensity` | `bool` | PBOD compatibility switch to zero the intensity channel. | - |
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

**Model Config Parameters**

| Parameter | Type | Description | Constraints |
| --- | --- | --- | --- |
| `intensity_threshold` | `int` | Saturation value for point intensity normalisation. | Must be ≥ 0. |
| `with_velocity` | `bool` | Whether the model predicts planar velocity. | - |
| `x_min` | `double` | Lower x-bound of the preprocessing region. | Must be finite and satisfy `x_min < x_max`. |
| `x_max` | `double` | Upper x-bound of the preprocessing region. | Must be finite and satisfy `x_min < x_max`. |
| `y_min` | `double` | Lower y-bound of the preprocessing region. | Must be finite and satisfy `y_min < y_max`. |
| `y_max` | `double` | Upper y-bound of the preprocessing region. | Must be finite and satisfy `y_min < y_max`. |
| `z_min` | `double` | Lower z-bound of the preprocessing region. | Must be finite and satisfy `z_min < z_max`. |
| `z_max` | `double` | Upper z-bound of the preprocessing region. | Must be finite and satisfy `z_min < z_max`. |
| `x_grid_size` | `int` | Number of grid cells along the x-axis. | Must be > 0. |
| `y_grid_size` | `int` | Number of grid cells along the y-axis. | Must be > 0. |
| `predicted_class_names` | `string array` | [**dynamic**] Class labels in network output order. | Must not be empty; each entry must be non-empty. |
| `model_bounds.publish_polygon` | `bool` | Publish the xy bounds as `geometry_msgs/msg/PolygonStamped` on `~/model_bounds`. | - |

**NMS parameters for PBOD and PP**

| Parameter | Type | Description | Constraints |
| --- | --- | --- | --- |
| `nms_max_num_objects` | `int` | [**dynamic**] Maximum number of objects considered during NMS. | Must be ≥ 0. |
| `nms_iou_threshold` | `double` | [**dynamic**] IoU threshold for suppression. | Must be within `[0.0, 1.0]`. |
| `nms_score_threshold` | `double array` | [**dynamic**] Minimum score(s) used during suppression. | Provide either one value or one per class; every entry must lie within `[0.0, 1.0]`. |

**PBOD / TPOD Model Config Parameters**

| Parameter | Type | Description | Constraints |
| --- | --- | --- | --- |
| `max_num_points` | `int` | Upper bound on points processed per cloud. | Must be > 0. |
| `stride` | `int array` | Strides of the three pillar blocks. | Must not be empty; every entry must be > 0. |
| `first_up_stride` | `int` | First upsample stride controlling output size. | Must be > 0. |

**TPOD-specific Parameter**

| Parameter | Type | Description | Constraints |
| --- | --- | --- | --- |
| `cls_threshold` | `float` | [**dynamic**] Minimum class probability required to keep a detection. | Must be within `[0.0, 1.0]`. |

**PP-specific Model Config Parameters**

| Parameter | Type | Description | Constraints |
| --- | --- | --- | --- |
| `max_pillars` | `int` | Maximum number of pillars. | Must be > 0. |
| `max_points_per_pillar` | `int` | Maximum number of points stored per pillar. | Must be > 0. |
| `n_features` | `int` | Number of features per augmented point. | Must be > 0. |
| `downscaling` | `int` | Spatial downscale factor. | Must be > 0. |
| `anchors_string` | `string` | All anchors described in a single parameter (optional) | Format: "[[length, width, height, z_center, yaw], [...]]" |
| `anchors.size` | `int` | Number of anchors. | Must be > 0. |
| `anchors.anchor_<i>` | `double array` | Anchor definition `[length, width, height, z_center, yaw]`. | Must contain exactly five values; length, width, and height must be > 0 and all values must be finite. |


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
| `point_cloud_object_detection` | `point_cloud_object_detection.launch.py` | `/docker-ros/ws/install/share/point_cloud_object_detection/launch` | Launch the inference node and before that, combine params.yml and slikaf config file |
| `point_cloud_object_detection` | `point_cloud_object_detection_component.launch.py` | `/docker-ros/ws/install/share/point_cloud_object_detection/launch` | Launch the inference node as a component and before that, combine params.yml and slikaf config file |
| `point_cloud_object_detection` | `point_cloud_object_detection_minimal.launch.py` | `/docker-ros/ws/install/share/point_cloud_object_detection/launch` | Launch the inference node with an existing combined config file |

### Configuration Files

| Package | File | Path | Description |
| --- | --- | --- | --- |
| `point_cloud_object_detection` | `params.yml` | `/docker-ros/ws/install/point_cloud_object_detection/share/point_cloud_object_detection/config` | File containing static parameters |
| `point_cloud_object_detection` | `config.yml` | `/docker-ros/ws/install/point_cloud_object_detection/share/point_cloud_object_detection/config` | File containing model config parameters |


## Point Cloud Input Fields

The node accepts `sensor_msgs/PointCloud2` messages that always contain XYZ coordinates. The `point_type` parameter controls how additional features are extracted before being forwarded to the model:

- `PointXYZI` *(default)* – consumes the ROS `intensity` field as the single feature channel. `num_point_features` **must** be set to 1 in this mode.
- `PointXYZRV` – expects floating-point `reflectivity` and `velocity` fields. Reflectivity replaces the classic intensity channel, and velocity is forwarded as the second feature. `num_point_features` must be set to 2. Missing reflectivity falls back to the message `intensity`, and missing velocity is treated as zero.

Additional feature channels beyond these presets are currently not supported and are filled with zeros when requested.
