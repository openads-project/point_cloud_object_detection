// Copyright Institute for Automotive Engineering (ika), RWTH Aachen University
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>
#include <limits>

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

inline constexpr std::array<const char*, 2> kAllowedPointFeatureFields = {"intensity", "reflectivity"};
inline constexpr std::array<const char*, 3> kSupportedManifestPrecisions = {"fp32", "fp16", "int8"};
inline constexpr std::array<const char*, 2> kAllowedPreprocessingBackends = {"cpu", "cuda"};

// model config
struct ModelConfig {
  // Preprocessing
  std::string preprocessing_backend = "cpu";
  std::string point_feature_normalization_type = "none";
  float point_feature_value_threshold = 0.0F;
  float point_feature_min_value = 0.0F;
  float point_feature_max_value = 0.0F;
  float point_feature_norm_epsilon = 1e-6F;

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
  float contract_z_min = 0.0F;
  float contract_z_max = 0.0F;
  int x_grid_size;
  int y_grid_size;
  float voxel_x = 0.0F;
  float voxel_y = 0.0F;
  float voxel_z = 0.0F;

  // NMS
  int nms_max_num_objects;
  float nms_iou_threshold;
  std::vector<double> nms_score_threshold;

  // PBOD specific config
  int max_num_points;
  std::vector<int64_t> stride;
  int first_up_stride;

  std::vector<int64_t> pillar_map_size;
  std::vector<std::vector<float>> pillar_map_range;

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
  double triton_client_timeout_s = 2.0;
  int64_t sensor_id = 0;

  // Exported runtime defaults from model_manifest.yml, overridable via ROS
  // parameters.
  double nms_iou_threshold = 0.0;
  int64_t nms_max_num_objects = 0;
  double output_class_score_threshold = 0.0;
  double point_feature_value_threshold = 0.0;

  // Optional no-detection rectangle (in inference_frame) where detections are
  // not allowed
  double no_detection_zone_x_min = 0.0;
  double no_detection_zone_x_max = 0.0;
  double no_detection_zone_y_min = 0.0;
  double no_detection_zone_y_max = 0.0;

  // Detection area (circular sector) in inference_frame where detections are
  // allowed
  double detection_area_center_x = 0.0;
  double detection_area_center_y = 0.0;
  double detection_area_radius = 0.0;
  double detection_area_bearing_deg = 0.0;  // central azimuth (deg, 0 along +x, CCW)
  double detection_area_fov_deg = 360.0;    // angular width (deg)
  // Runtime override for the effective z range used during inference-time
  // preprocessing. Defaults to the manifest z range.
  double detection_area_z_min = 0.0;
  double detection_area_z_max = 0.0;

  // Auxiliary grid-map publications
  double density_grid_map_gain = 1.0;
  double dynamic_grid_map_gain = 1.0;
  double combined_grid_map_gain = 1.0;
  double static_grid_map_gain = 1.0;

  std::vector<double> variance = std::vector<double>(12, -1.0);  // CONTINUOUS_STATE_COVARIANCE_UNKNOWN sentinel
  std::vector<double> nms_score_threshold;

  std::string preprocessing_backend = "cpu";
  std::string model_repository;
  std::string model_name;
  std::string model_version;
  std::string server_url;       // required
  std::string inference_frame;  // required
  std::string output_frame;     // required
  std::string point_feature_field = "intensity";
  std::string detection_area_filter_mode = "center";  // "center" or "complete"
  std::string grid_map_frame;

  int detection_area_num_segments = 32;  // polygon approximation of arc

  bool use_shm = false;
  bool cuda_input_shm = false;

  bool no_detection_zone_enabled = false;
  // If true, remove raw points in the no-detection zone from model input
  bool no_detection_zone_remove_points = false;
  // If true, publish the bounds of the no-detection zone as a PolygonStamped
  bool no_detection_zone_publish_polygon = false;
  // If true, publish raw points that lie inside the no-detection zone
  bool no_detection_zone_publish_points = false;

  bool detection_area_enabled = false;  // turns on sector-based filtering/publishing
  bool detection_area_publish_polygon = false;
  // Detection filtering mode
  // If enabled, remove detections that lie outside the area either by center or
  // completely
  bool detection_area_filter_detections = false;  // enable detection filtering by sector

  // Model bounds polygon publication (XY rectangle from
  // x_min/x_max/y_min/y_max)
  bool model_bounds_publish_polygon = false;

  bool publish_density_grid_map = false;
  bool publish_dynamic_grid_map = false;
  bool publish_combined_grid_map = false;
  bool publish_static_grid_map = false;
  bool zero_grid_map_cells_in_no_detection_zone = false;
  bool zero_grid_map_cells_outside_detection_area = false;
};

}  // namespace point_cloud_object_detection
