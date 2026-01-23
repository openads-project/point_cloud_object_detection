#pragma once

#include <tf2/transform_datatypes.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include <Eigen/Dense>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2_sensor_msgs/tf2_sensor_msgs.hpp>

#include "pcod_common/bounding_box.hpp"

namespace point_cloud_object_detection {
using pcod_common::BoundingBox;
using pcod_common::ClassificationEntry;

// model config
struct ModelConfig {
  // Preprocessing
  std::string point_feature_normalization_type = "none";
  float point_feature_intensity_threshold = 0.0f;
  float point_feature_min_intensity = 0.0f;
  float point_feature_max_intensity = 0.0f;
  float point_feature_norm_epsilon = 1e-6f;

  // Postprocessing
  std::vector<std::string> predicted_class_names;
  std::map<std::string, uint8_t> class_mapping_;

  // Grid
  float x_min;
  float x_max;
  float y_min;
  float y_max;
  float z_min;
  float z_max;
  int x_grid_size;
  int y_grid_size;
  float voxel_x = 0.0f;
  float voxel_y = 0.0f;
  float voxel_z = 0.0f;

  // NMS
  int nms_max_num_objects;
  float nms_iou_threshold;
  std::vector<double> nms_score_threshold;

  // PBOD specific config
  int max_num_points;
  int num_point_features = 1;
  std::vector<int64_t> stride;
  int first_up_stride;

  std::vector<int64_t> pillar_map_size;
  std::vector<std::vector<float>> pillar_map_range;

  // Legacy compatibility
  bool mask_is_bool;
  bool zero_intensity;

  // No-detection zone point filtering (in inference_frame)
  bool no_detection_zone_remove_points = false;  // If true, drop raw points in the zone from model input
  double no_detection_zone_x_min = 0.0;
  double no_detection_zone_x_max = 0.0;
  double no_detection_zone_y_min = 0.0;
  double no_detection_zone_y_max = 0.0;

  // Detection area (circular sector) point filtering (in inference_frame)
  // If true, drop raw points outside the sector from model input
  bool detection_area_remove_points_outside = false;
  double detection_area_center_x = 0.0;     // sector center x (m)
  double detection_area_center_y = 0.0;     // sector center y (m)
  double detection_area_radius = 0.0;       // sector radius (m)
  double detection_area_bearing_deg = 0.0;  // sector central azimuth (deg, 0 along +x, CCW positive)
  double detection_area_fov_deg = 360.0;    // sector FOV angle (deg)
};

// parameters
struct Params {
  std::string model_name;
  std::string model_version;
  std::string server_url;
  bool use_shm;
  std::string model_manifest_path;

  std::string inference_frame;
  std::string output_frame;

  uint64_t sensor_id;

  std::vector<double> variance;
  double class_score_threshold;

  // Optional no-detection rectangle (in inference_frame) where detections are not allowed
  bool no_detection_zone_enabled = false;
  double no_detection_zone_x_min = 0.0;
  double no_detection_zone_x_max = 0.0;
  double no_detection_zone_y_min = 0.0;
  double no_detection_zone_y_max = 0.0;
  // If true, publish the bounds of the no-detection zone as a PolygonStamped
  bool no_detection_zone_publish_polygon = false;
  // If true, publish raw points that lie inside the no-detection zone
  bool no_detection_zone_publish_points = false;

  // Detection area (circular sector) in inference_frame where detections are allowed
  bool detection_area_enabled = false;  // turns on sector-based filtering/publishing
  double detection_area_center_x = 0.0;
  double detection_area_center_y = 0.0;
  double detection_area_radius = 0.0;
  double detection_area_bearing_deg = 0.0;  // central azimuth (deg, 0 along +x, CCW)
  double detection_area_fov_deg = 360.0;    // angular width (deg)
  bool detection_area_publish_polygon = false;
  int detection_area_num_segments = 32;  // polygon approximation of arc

  // Detection filtering mode
  // If enabled, remove detections that lie outside the area either by center or completely
  bool detection_area_filter_detections = false;      // enable detection filtering by sector
  std::string detection_area_filter_mode = "center";  // "center" or "complete"

  // Model bounds polygon publication (XY rectangle from x_min/x_max/y_min/y_max)
  bool model_bounds_publish_polygon = false;
};

}  // namespace point_cloud_object_detection
