#include "point_cloud_object_detection/PointCloudObjectDetection.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iostream>
#include <memory>
#include <sstream>
#include <utility>

#include <boost/algorithm/string/case_conv.hpp>

#include <rclcpp/logging.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <sensor_msgs/msg/point_field.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <stdexcept>

#include "point_cloud_object_detection/Utils.hpp"

namespace point_cloud_object_detection {

namespace {

[[noreturn]] void throwParameterError(const rclcpp::Logger& logger, const std::string& param_name,
                                      const std::string& details) {
  RCLCPP_FATAL(logger, "Invalid parameter '%s': %s", param_name.c_str(), details.c_str());
  throw std::runtime_error("Invalid parameter '" + param_name + "': " + details);
}

bool isProbability(double value) { return std::isfinite(value) && value >= 0.0 && value <= 1.0; }

bool isFinite(double value) { return std::isfinite(value); }

void sanitizeVarianceVector(std::vector<double>& variance, double sentinel) {
  for (double& value : variance) {
    if (value < 0.0 && std::fabs(value + 1.0) <= 1e-9) {
      value = sentinel;
    }
  }
}

PointType parsePointType(const std::string& value, const rclcpp::Logger& logger) {
  std::string lowered = boost::algorithm::to_lower_copy(value);
  if (lowered == "pointxyzi") {
    return PointType::XYZI;
  }
  if (lowered == "pointxyzrv") {
    return PointType::XYZRV;
  }
  throwParameterError(logger, "point_type", "must be 'PointXYZI' or 'PointXYZRV'");
  return PointType::XYZI;
}

}  // namespace

// constants
const std::string PointCloudObjectDetection::kInputTopic = "~/point_cloud";
const std::string PointCloudObjectDetection::kOutputTopic = "~/object_list";
const std::string PointCloudObjectDetection::kClassPointCloudsTopicBase = "~/class_point_cloud/";
const std::string PointCloudObjectDetection::kUnclassifiedPointsTopic = "~/unclassified_point_cloud";
const std::string PointCloudObjectDetection::kUnclassifiedOutsideAreaTopic =
    "~/unclassified_points_outside_detection_area";
const std::string PointCloudObjectDetection::kNoDetectionZoneTopic = "~/no_detection_zone";
const std::string PointCloudObjectDetection::kNoDetectionZonePointsTopic = "~/no_detection_zone_points";
const std::string PointCloudObjectDetection::kDetectionAreaTopic = "~/detection_area";
const std::string PointCloudObjectDetection::kModelBoundsTopic = "~/model_bounds";
const std::map<uint8_t, std::vector<std::string>> PointCloudObjectDetection::kPossibleClassNames{
    {pm::msg::ObjectClassification::CAR, {"car", "vehicle"}},
    {pm::msg::ObjectClassification::PEDESTRIAN, {"pedestrian", "human", "man", "woman", "person"}},
    {pm::msg::ObjectClassification::BICYCLE, {"bicycle", "bike", "cyclist"}},
    {pm::msg::ObjectClassification::MOTORBIKE, {"motorcycle", "motorbike"}},
    {pm::msg::ObjectClassification::TRUCK, {"truck"}},
    {pm::msg::ObjectClassification::BUS, {"bus"}},
    {pm::msg::ObjectClassification::VAN, {"van"}},
    {pm::msg::ObjectClassification::ANIMAL, {"animal"}},
    {pm::msg::ObjectClassification::ROAD_OBSTACLE, {"obstacle", "road_obstacle"}},
    {pm::msg::ObjectClassification::TRAIN, {"train"}},
    {pm::msg::ObjectClassification::TRAILER, {"trailer"}}};

PointCloudObjectDetection::PointCloudObjectDetection(const rclcpp::NodeOptions& options)
    : rclcpp::Node("point_cloud_object_detection", options) {
  declareParameters();
  loadParameters();

  // run setup after constructor has finished to enable shared_from_this()
  setup_timer_ = this->create_wall_timer(std::chrono::milliseconds(1), [this]() {
    setup();
    setup_timer_->cancel();
  });
}

template <typename T>
void PointCloudObjectDetection::declare_parameter_if_not_exists(const std::string& name, const T& type_or_default,
                                                                const std::string& desc) {
  auto params = list_parameters({name}, 1);
  if (std::find(params.names.begin(), params.names.end(), name) == params.names.end()) {
    rcl_interfaces::msg::ParameterDescriptor descriptor;
    descriptor.description = desc;
    declare_parameter(name, type_or_default, descriptor);
  }
}

void PointCloudObjectDetection::declareParameters() {
  declare_parameter_if_not_exists("model_name", rclcpp::ParameterType::PARAMETER_STRING, "Model name on Triton server");
  declare_parameter_if_not_exists("model_version", rclcpp::ParameterType::PARAMETER_STRING,
                                  "Model version on Triton server");
  declare_parameter_if_not_exists("server_url", rclcpp::ParameterType::PARAMETER_STRING,
                                  "URL of the triton server, e.g. 134.130.20.221:8001");
  declare_parameter_if_not_exists("use_shm", false, "Whether or not to use shared memory for Triton");
  declare_parameter_if_not_exists("inference_frame", "", "Frame for inference");
  declare_parameter_if_not_exists("output_frame", "", "Frame for object list");
  declare_parameter_if_not_exists("sensor_id", -1, "Sensor ID for object list");

  declare_parameter_if_not_exists(
      "variances", rclcpp::ParameterType::PARAMETER_DOUBLE_ARRAY,
      "Array with variances. Entries correspond to ISCACTR model defined in perception interfaces");

  declare_parameter_if_not_exists("class_score_threshold", 0.0, "Model config: Class score threshold");

  declare_parameter_if_not_exists("intensity_threshold", 10000, "Model config: Intensity threshold");
  declare_parameter_if_not_exists("with_velocity", false, "Model config: with velocity");

  declare_parameter_if_not_exists("x_min", -40.96, "Model config: x_min for Grid");
  declare_parameter_if_not_exists("x_max", 40.96, "Model config: x_max for Grid");
  declare_parameter_if_not_exists("y_min", 0.0, "Model config: y_min for Grid");
  declare_parameter_if_not_exists("y_max", 40.96, "Model config: y_max for Grid");
  declare_parameter_if_not_exists("z_min", -1.0, "Model config: z_min for Grid");
  declare_parameter_if_not_exists("z_max", 5.0, "Model config: z_max for Grid");
  declare_parameter_if_not_exists("x_grid_size", 512, "Model config: x_grid_size for Grid");
  declare_parameter_if_not_exists("y_grid_size", 256, "Model config: y_grid_size for Grid");

  declare_parameter_if_not_exists(
      "predicted_class_names", rclcpp::ParameterType::PARAMETER_STRING_ARRAY,
      "Array with class names. Each class corresponds to an entry in the model class prediction "
      "output, and should match case-insensitive one of the names in perception_interfaces.");

  declare_parameter_if_not_exists("publish_class_point_clouds", false,
                                  "Whether to publish point clouds for each detected class");

  declare_parameter_if_not_exists("publish_unclassified_points", false,
                                  "Whether to publish points that are not inside any bounding box");
  declare_parameter_if_not_exists(
      "publish_unclassified_points_outside_detection_area", false,
      "Publish unclassified points outside the detection area (excluding no-detection zone)");

  // No-detection zone (inference_frame) where detections are not allowed
  declare_parameter_if_not_exists("no_detection_zone.enabled", false,
                                  "Enable rectangular no-detection zone in inference_frame");
  declare_parameter_if_not_exists("no_detection_zone.remove_points", false,
                                  "If true, remove raw points inside the no-detection zone from model input and"
                                  " unclassified point publishing");
  declare_parameter_if_not_exists("no_detection_zone.x_min", 0.0, "No-detection zone x_min (inference_frame)");
  declare_parameter_if_not_exists("no_detection_zone.x_max", 0.0, "No-detection zone x_max (inference_frame)");
  declare_parameter_if_not_exists("no_detection_zone.y_min", 0.0, "No-detection zone y_min (inference_frame)");
  declare_parameter_if_not_exists("no_detection_zone.y_max", 0.0, "No-detection zone y_max (inference_frame)");
  declare_parameter_if_not_exists("no_detection_zone.publish_polygon", false,
                                  "If true, publish a geometry_msgs/PolygonStamped with the no-detection zone bounds");
  declare_parameter_if_not_exists("no_detection_zone.publish_points", false,
                                  "If true, publish raw points inside the no-detection zone");

  // Detection area (circular sector in inference_frame)
  declare_parameter_if_not_exists("detection_area.enabled", false, "Enable circular-sector detection area");
  declare_parameter_if_not_exists("detection_area.center_x", 0.0, "Detection area center x (m) in inference_frame");
  declare_parameter_if_not_exists("detection_area.center_y", 0.0, "Detection area center y (m) in inference_frame");
  declare_parameter_if_not_exists("detection_area.radius", 0.0, "Detection area radius (m)");
  declare_parameter_if_not_exists("detection_area.bearing_deg", 0.0,
                                  "Detection area central azimuth (deg, 0 along +x, CCW positive)");
  declare_parameter_if_not_exists("detection_area.fov_deg", 360.0, "Detection area FOV angle (deg)");
  declare_parameter_if_not_exists("detection_area.publish_polygon", false,
                                  "Publish geometry_msgs/PolygonStamped approximating the sector");
  declare_parameter_if_not_exists("detection_area.num_segments", 32,
                                  "Number of segments to approximate the circular arc (>= 3)");
  declare_parameter_if_not_exists("detection_area.filter_detections", false,
                                  "Remove detections outside the detection area");
  declare_parameter_if_not_exists("detection_area.filter_mode", std::string("center"),
                                  "Filtering mode: 'center' or 'complete'");

  // Publish model XY bounds as polygon
  declare_parameter_if_not_exists("model_bounds.publish_polygon", false,
                                  "Publish the model x/y range rectangle as geometry_msgs/PolygonStamped");

  // NMS parameters
  declare_parameter_if_not_exists("nms_max_num_objects", 50, "Model config: Max number of detected objects for NMS");
  declare_parameter_if_not_exists("nms_iou_threshold", 0.1, "Model config: IoU-threshold for NMS");
  declare_parameter_if_not_exists("nms_score_threshold", std::vector<double>{0.2},
                                  "Model config: Min cell score/occupancy for NMS");

  // PBOD & TPOD specific params
  declare_parameter_if_not_exists("max_num_points", 140000, "Model config: Max number of points");
  declare_parameter_if_not_exists("num_point_features", 1, "Model config: number of point features per point");
  declare_parameter_if_not_exists("point_type", std::string("PointXYZI"),
                                  "Point struct to derive features from (PointXYZI or PointXYZRV)");
  declare_parameter_if_not_exists("stride", rclcpp::ParameterType::PARAMETER_INTEGER_ARRAY, "Array with strides");
  declare_parameter_if_not_exists("first_up_stride", 1, "Model config: first up stride");

  // PBOD & TPOD specific params/Legacy compatibility
  declare_parameter_if_not_exists("mask_is_bool", true, "Whether the mask is a boolean or a float mask");
  declare_parameter_if_not_exists("zero_intensity", false, "Whether to set intensity to zero for all points");

  // Only for TPOD
  declare_parameter_if_not_exists("cls_threshold", 0.3, "Model config: Class score threshold for TPOD");

  // PP specific params
  declare_parameter_if_not_exists("max_pillars", 10000, "Model config: Max pillars");
  declare_parameter_if_not_exists("max_points_per_pillar", 100, "Model config: Max points per pillar");
  declare_parameter_if_not_exists("n_features", 9, "Model config: number of features");
  declare_parameter_if_not_exists("downscaling", 2, "Model config: downscaling");
  declare_parameter_if_not_exists("anchors.size", 8, "Model config: number of anchors");
  declare_parameter_if_not_exists(
      "anchors_string", std::string(""),
      "Model config: all anchors as JSON string array [[l,w,h,z,yaw], ...] (alternative to numbered anchors)");
}

void PointCloudObjectDetection::loadParameters() {
  // model path
  try {
    params_.model_name = this->get_parameter("model_name").as_string();
    params_.model_version = this->get_parameter("model_version").as_string();
    params_.server_url = this->get_parameter("server_url").as_string();
  } catch (rclcpp::exceptions::ParameterUninitializedException&) {
    RCLCPP_FATAL(this->get_logger(), "Parameters 'model_name', 'model_version', and 'server_url' are required");
    exit(EXIT_FAILURE);
  }
  params_.use_shm = this->get_parameter("use_shm").as_bool();

  // inference frame
  params_.inference_frame = get_parameter("inference_frame").as_string();

  // output frame
  params_.output_frame = get_parameter("output_frame").as_string();

  // sensor id
  const int64_t sensor_id = get_parameter("sensor_id").as_int();
  if (sensor_id < -1) {
    throwParameterError(this->get_logger(), "sensor_id", "must be -1 (random) or a non-negative integer");
  }
  params_.sensor_id = sensor_id == -1 ? static_cast<uint64_t>(random()) : static_cast<uint64_t>(sensor_id);

  // variances
  auto cscu = pm::object_access::CONTINUOUS_STATE_COVARIANCE_UNKNOWN;
  get_parameter_or("variances", params_.variance,
                   {cscu, cscu, cscu, cscu, cscu, cscu, cscu, cscu, cscu, cscu, cscu, cscu});
  sanitizeVarianceVector(params_.variance, cscu);

  // class score threshold
  params_.class_score_threshold = get_parameter("class_score_threshold").as_double();

  // class point cloud publishing
  params_.publish_class_point_clouds = get_parameter("publish_class_point_clouds").as_bool();

  // unclassified points publishing
  params_.publish_unclassified_points = get_parameter("publish_unclassified_points").as_bool();
  params_.publish_unclassified_points_outside_detection_area =
      get_parameter("publish_unclassified_points_outside_detection_area").as_bool();

  // no-detection zone parameters
  params_.no_detection_zone_enabled = get_parameter("no_detection_zone.enabled").as_bool();
  params_.no_detection_zone_x_min = get_parameter("no_detection_zone.x_min").as_double();
  params_.no_detection_zone_x_max = get_parameter("no_detection_zone.x_max").as_double();
  params_.no_detection_zone_y_min = get_parameter("no_detection_zone.y_min").as_double();
  params_.no_detection_zone_y_max = get_parameter("no_detection_zone.y_max").as_double();
  params_.no_detection_zone_publish_polygon = get_parameter("no_detection_zone.publish_polygon").as_bool();
  params_.no_detection_zone_publish_points = get_parameter("no_detection_zone.publish_points").as_bool();

  // detection area parameters
  params_.detection_area_enabled = get_parameter("detection_area.enabled").as_bool();
  params_.detection_area_center_x = get_parameter("detection_area.center_x").as_double();
  params_.detection_area_center_y = get_parameter("detection_area.center_y").as_double();
  params_.detection_area_radius = get_parameter("detection_area.radius").as_double();
  params_.detection_area_bearing_deg = get_parameter("detection_area.bearing_deg").as_double();
  params_.detection_area_fov_deg = get_parameter("detection_area.fov_deg").as_double();
  params_.detection_area_publish_polygon = get_parameter("detection_area.publish_polygon").as_bool();
  params_.detection_area_num_segments = get_parameter("detection_area.num_segments").as_int();
  params_.detection_area_filter_detections = get_parameter("detection_area.filter_detections").as_bool();
  params_.detection_area_filter_mode = get_parameter("detection_area.filter_mode").as_string();

  // model bounds polygon
  params_.model_bounds_publish_polygon = get_parameter("model_bounds.publish_polygon").as_bool();

  validateParamsOrThrow();
}

void PointCloudObjectDetection::validateParamsOrThrow() const {
  auto fail = [this](const std::string& param, const std::string& message) {
    throwParameterError(this->get_logger(), param, message);
  };

  const double cscu = pm::object_access::CONTINUOUS_STATE_COVARIANCE_UNKNOWN;
  if (params_.variance.size() != 12) {
    fail("variances", "must contain 12 entries corresponding to the ISCACTR state");
  }
  for (std::size_t i = 0; i < params_.variance.size(); ++i) {
    const double value = params_.variance[i];
    if (!isFinite(value)) {
      fail("variances", "all entries must be finite");
    }
    if (value < 0.0 && std::fabs(value - cscu) > 1e-9) {
      fail("variances", "entries must be non-negative or equal to CONTINUOUS_STATE_COVARIANCE_UNKNOWN (-1)");
    }
  }

  if (!isProbability(params_.class_score_threshold)) {
    fail("class_score_threshold", "must be within [0.0, 1.0]");
  }

  if (!isFinite(params_.no_detection_zone_x_min) || !isFinite(params_.no_detection_zone_x_max) ||
      !isFinite(params_.no_detection_zone_y_min) || !isFinite(params_.no_detection_zone_y_max)) {
    fail("no_detection_zone", "bounds must be finite values");
  }
  if (params_.no_detection_zone_enabled) {
    if (!(params_.no_detection_zone_x_min < params_.no_detection_zone_x_max) ||
        !(params_.no_detection_zone_y_min < params_.no_detection_zone_y_max)) {
      fail("no_detection_zone", "requires x_min < x_max and y_min < y_max when enabled");
    }
  }

  if (!isFinite(params_.detection_area_center_x) || !isFinite(params_.detection_area_center_y)) {
    fail("detection_area.center", "center_x and center_y must be finite");
  }
  if (!isFinite(params_.detection_area_radius)) {
    fail("detection_area.radius", "must be finite");
  }
  if (params_.detection_area_radius < 0.0) {
    fail("detection_area.radius", "must be non-negative");
  }
  if (!isFinite(params_.detection_area_fov_deg)) {
    fail("detection_area.fov_deg", "must be finite");
  }
  if (params_.detection_area_fov_deg <= 0.0 || params_.detection_area_fov_deg > 360.0) {
    fail("detection_area.fov_deg", "must be in the range (0, 360]");
  }
  if (!isFinite(params_.detection_area_bearing_deg)) {
    fail("detection_area.bearing_deg", "must be finite");
  }
  if (params_.detection_area_bearing_deg < -360.0 || params_.detection_area_bearing_deg > 360.0) {
    fail("detection_area.bearing_deg", "must be within [-360, 360]");
  }
  if (params_.detection_area_num_segments < 3) {
    fail("detection_area.num_segments", "must be greater than or equal to 3");
  }
  if (params_.detection_area_enabled && params_.detection_area_radius <= 0.0) {
    fail("detection_area.radius", "must be greater than 0 when detection_area.enabled is true");
  }
  if (params_.detection_area_filter_mode != "center" && params_.detection_area_filter_mode != "complete") {
    fail("detection_area.filter_mode", "must be either 'center' or 'complete'");
  }
}

bool PointCloudObjectDetection::updateNMSScoreThreshold(std::vector<double>& score_thresholds) {
  if (score_thresholds.size() == 1) {
    for (std::size_t i = 1; i < model_config_.predicted_class_names.size(); i++) {
      // If a single NMS score threshold is provided, replicate it for each class.
      score_thresholds.push_back(score_thresholds[0]);
    }
  }
  if (score_thresholds.size() != model_config_.predicted_class_names.size()) {
    RCLCPP_ERROR(this->get_logger(), "The number of NMS score thresholds does not match the number of classes.");
    return false;
  }
  model_config_.nms_score_threshold = score_thresholds;
  model_config_.min_nms_score_threshold =
      *std::min_element(model_config_.nms_score_threshold.begin(), model_config_.nms_score_threshold.end());
  return true;
}

void PointCloudObjectDetection::loadModelConfig() {
  // Only execute during initialization of the node and not for parameter changes during runtime
  if (!model_runtime_overwrite_) {
    // intensity threshold
    model_config_.intensity_threshold = get_parameter("intensity_threshold").as_int();

    // grid parameters
    model_config_.x_min = get_parameter("x_min").as_double();
    model_config_.x_max = get_parameter("x_max").as_double();
    model_config_.y_min = get_parameter("y_min").as_double();
    model_config_.y_max = get_parameter("y_max").as_double();
    model_config_.z_min = get_parameter("z_min").as_double();
    model_config_.z_max = get_parameter("z_max").as_double();
    model_config_.x_grid_size = get_parameter("x_grid_size").as_int();
    model_config_.y_grid_size = get_parameter("y_grid_size").as_int();

    // with velocity or not
    model_config_.with_velocity = get_parameter("with_velocity").as_bool();
    get_parameter_or("predicted_class_names", model_config_.predicted_class_names,
                     {
                         "UNCLASSIFIED",
                         "PEDESTRIAN",
                         "BICYCLE",
                         "MOTORBIKE",
                         "CAR",
                         "TRUCK",
                         "VAN",
                         "BUS",
                         "ANIMAL",
                         "ROAD_OBSTACLE",
                         "TRAIN",
                         "TRAILER",
                     });
  }

  for (auto& name : model_config_.predicted_class_names) {
    boost::algorithm::to_lower(name);
    uint8_t pm_type = pm::msg::ObjectClassification::UNCLASSIFIED;
    for (auto& [type, possible_names] : kPossibleClassNames) {
      // If the name is in the possible names for any class in perception_msgs, assign it.
      if (std::find(possible_names.begin(), possible_names.end(), name) != possible_names.end()) {
        pm_type = type;
        break;
      }
    }
    // Warn if class is unknown
    if (pm_type == pm::msg::ObjectClassification::UNCLASSIFIED) {
      RCLCPP_WARN_STREAM(get_logger(), "The class \"" << name << "\" is not mapped to any class of"
                                                      << " perception_msgs::msg::ObjectClassification");
    }
    model_config_.class_mapping_[name] = pm_type;
  }

  if ((model_type_ == ModelType::PBOD || model_type_ == ModelType::PP) && !model_runtime_overwrite_) {
    model_config_.nms_max_num_objects = get_parameter("nms_max_num_objects").as_int();
    if (model_config_.nms_max_num_objects < 0) {
      throwParameterError(this->get_logger(), "nms_max_num_objects", "cannot be negative");
    }
    model_config_.nms_iou_threshold = get_parameter("nms_iou_threshold").as_double();
    std::vector<double> nms_score_threshold = get_parameter("nms_score_threshold").as_double_array();
    if (!updateNMSScoreThreshold(nms_score_threshold)) {
      throwParameterError(this->get_logger(), "nms_score_threshold",
                          "must contain exactly one value or one per predicted class");
    }
  }

  // PBOD & TPOD specific config loading
  if ((model_type_ == ModelType::PBOD || model_type_ == ModelType::TPOD)) {
    if (!updateNMSScoreThreshold(model_config_.nms_score_threshold)) {
      throwParameterError(this->get_logger(), "nms_score_threshold",
                          "must contain exactly one value or one per predicted class");
    }

    // setup pillar map size
    model_config_.pillar_map_size = {model_config_.x_grid_size, model_config_.y_grid_size};

    // setup pillar map range
    model_config_.pillar_map_range = {{model_config_.x_min, model_config_.x_max},
                                      {model_config_.y_min, model_config_.y_max},
                                      {model_config_.z_min, model_config_.z_max}};

    if (model_config_.no_detection_zone_remove_points) {
      const bool valid_bounds = model_config_.no_detection_zone_x_min < model_config_.no_detection_zone_x_max &&
                                model_config_.no_detection_zone_y_min < model_config_.no_detection_zone_y_max;
      if (!valid_bounds) {
        throwParameterError(this->get_logger(), "no_detection_zone.remove_points",
                            "requires x_min < x_max and y_min < y_max");
      }
    }

    if (!model_runtime_overwrite_) {
      model_config_.max_num_points = get_parameter("max_num_points").as_int();
      model_config_.num_point_features = get_parameter("num_point_features").as_int();
      model_config_.point_type = get_parameter("point_type").as_string();
      point_type_ = parsePointType(model_config_.point_type, this->get_logger());
      get_parameter_or("stride", model_config_.stride, {2, 1, 2});
      model_config_.first_up_stride = get_parameter("first_up_stride").as_int();

      model_config_.mask_is_bool = get_parameter("mask_is_bool").as_bool();
      model_config_.zero_intensity = get_parameter("zero_intensity").as_bool();

      // Copy no-detection zone settings for point filtering into model_config
      // Note: bounds are defined in the inference_frame
      model_config_.no_detection_zone_remove_points = get_parameter("no_detection_zone.remove_points").as_bool();
      model_config_.no_detection_zone_x_min = get_parameter("no_detection_zone.x_min").as_double();
      model_config_.no_detection_zone_x_max = get_parameter("no_detection_zone.x_max").as_double();
      model_config_.no_detection_zone_y_min = get_parameter("no_detection_zone.y_min").as_double();
      model_config_.no_detection_zone_y_max = get_parameter("no_detection_zone.y_max").as_double();

      // Detection area -> point filtering in model input
      model_config_.detection_area_remove_points_outside =
          get_parameter("detection_area.enabled").as_bool() &&
          get_parameter("detection_area.radius").as_double() > 0.0 &&
          get_parameter("detection_area.fov_deg").as_double() > 0.0 &&
          get_parameter("detection_area.num_segments").as_int() >= 3;  // segments checked for validity as proxy
      model_config_.detection_area_center_x = get_parameter("detection_area.center_x").as_double();
      model_config_.detection_area_center_y = get_parameter("detection_area.center_y").as_double();
      model_config_.detection_area_radius = get_parameter("detection_area.radius").as_double();
      model_config_.detection_area_bearing_deg = get_parameter("detection_area.bearing_deg").as_double();
      model_config_.detection_area_fov_deg = get_parameter("detection_area.fov_deg").as_double();

      if (model_type_ == ModelType::TPOD) {
        // Only needed for TPOD
        model_config_.cls_threshold = get_parameter("cls_threshold").as_double();
      }
    }

  }
  // PP specific config loading
  else if (model_type_ == ModelType::PP) {
    if (!model_runtime_overwrite_) {
      model_config_.max_pillars = get_parameter("max_pillars").as_int();
      model_config_.max_points_per_pillar = get_parameter("max_points_per_pillar").as_int();
      model_config_.n_features = get_parameter("n_features").as_int();
      model_config_.downscaling = get_parameter("downscaling").as_int();
      // Load Anchors
      const std::size_t n_anchors = get_parameter("anchors.size").as_int();
      if (n_anchors == 0) {
        throwParameterError(this->get_logger(), "anchors.size", "must be greater than zero");
      }

      model_config_.anchor_boxes.clear();
      model_config_.anchor_diagonals.clear();

      for (std::size_t i = 0; i < n_anchors; i++) {
        Anchor anchor;
        std::string description = "Model config: Anchor " + std::to_string(i);
        std::string param_name = "anchors.anchor_" + std::to_string(i);
        declare_parameter_if_not_exists(param_name, std::vector<double>{0.0, 0.0, 0.0, 0.0, 0.0}, description);
        std::vector<double> anchor_double = get_parameter(param_name).as_double_array();
        if (anchor_double.size() != 5) {
          throwParameterError(this->get_logger(), param_name, "must contain [length, width, height, z_center, yaw]");
        }
        anchor.length = anchor_double[0];
        anchor.width = anchor_double[1];
        anchor.height = anchor_double[2];
        anchor.z_center = anchor_double[3];
        anchor.yaw = anchor_double[4];

        model_config_.anchor_boxes.push_back(anchor);
        model_config_.anchor_diagonals.emplace_back(std::sqrt(std::pow(anchor.length, 2) + std::pow(anchor.width, 2)));
      }
    }

    model_config_.delta_x = (model_config_.x_max - model_config_.x_min) / model_config_.x_grid_size;
    model_config_.delta_y = (model_config_.y_max - model_config_.y_min) / model_config_.y_grid_size;

    if (!updateNMSScoreThreshold(model_config_.nms_score_threshold)) {
      throwParameterError(this->get_logger(), "nms_score_threshold",
                          "must contain exactly one value or one per predicted class");
    }

    if (!model_config_.anchors_string.empty()) {
      RCLCPP_INFO(this->get_logger(),
                  "Anchors String is not empty, trying to parse anchors from anchors_string parameter");
      std::string anchors_string = model_config_.anchors_string;
      model_config_.anchor_boxes.clear();
      model_config_.anchor_diagonals.clear();
      // Parse JSON array of arrays: "[[3.9, 1.6, 1.56, -1.0, 0.0], [3.9, 1.6, 1.56, -1.0, 1.5708], ...]"
      try {
        // Simple JSON parser for array of arrays
        std::vector<std::vector<double>> parsed_anchors;

        // Remove whitespace
        anchors_string.erase(std::remove_if(anchors_string.begin(), anchors_string.end(), ::isspace),
                             anchors_string.end());

        // Check for outer brackets
        if (anchors_string.front() == '[' && anchors_string.back() == ']') {
          anchors_string = anchors_string.substr(1, anchors_string.length() - 2);

          // Parse each inner array
          size_t pos = 0;
          while (pos < anchors_string.length()) {
            if (anchors_string[pos] == '[') {
              size_t end_bracket = anchors_string.find(']', pos);
              if (end_bracket == std::string::npos) {
                throw std::runtime_error("Malformed JSON: missing closing bracket");
              }

              std::string inner_array = anchors_string.substr(pos + 1, end_bracket - pos - 1);
              std::vector<double> anchor_values;

              // Parse comma-separated values
              std::istringstream iss(inner_array);
              std::string value_str;
              while (std::getline(iss, value_str, ',')) {
                if (!value_str.empty()) {
                  anchor_values.push_back(std::stod(value_str));
                }
              }

              if (anchor_values.size() == 5) {
                parsed_anchors.push_back(anchor_values);
              } else {
                throw std::runtime_error(
                    "Each anchor must have exactly 5 values [length, width, height, z_center, yaw]");
              }

              pos = end_bracket + 1;
              // Skip comma if present
              if (pos < anchors_string.length() && anchors_string[pos] == ',') {
                pos++;
              }
            } else {
              pos++;
            }
          }
        } else {
          throw std::runtime_error("anchors_string must be a JSON array of arrays");
        }

        // Convert parsed anchors to Anchor structs
        if (!parsed_anchors.empty()) {
          for (const auto& anchor_values : parsed_anchors) {
            Anchor anchor;
            anchor.length = anchor_values[0];
            anchor.width = anchor_values[1];
            anchor.height = anchor_values[2];
            anchor.z_center = anchor_values[3];
            anchor.yaw = anchor_values[4];

            model_config_.anchor_boxes.push_back(anchor);
            model_config_.anchor_diagonals.emplace_back(
                std::sqrt(std::pow(anchor.length, 2) + std::pow(anchor.width, 2)));
          }
          RCLCPP_INFO(this->get_logger(), "Loaded %zu anchors from anchors_string parameter", parsed_anchors.size());
        }
      } catch (const std::exception& e) {
        RCLCPP_WARN(this->get_logger(),
                    "Failed to parse anchors_string parameter: %s. Falling back to numbered parameters.", e.what());
      }
    }

    // Detection area -> point filtering in model input (also relevant for PP)
    model_config_.detection_area_remove_points_outside = get_parameter("detection_area.enabled").as_bool() &&
                                                         get_parameter("detection_area.radius").as_double() > 0.0 &&
                                                         get_parameter("detection_area.fov_deg").as_double() > 0.0 &&
                                                         get_parameter("detection_area.num_segments").as_int() >= 3;
    model_config_.detection_area_center_x = get_parameter("detection_area.center_x").as_double();
    model_config_.detection_area_center_y = get_parameter("detection_area.center_y").as_double();
    model_config_.detection_area_radius = get_parameter("detection_area.radius").as_double();
    model_config_.detection_area_bearing_deg = get_parameter("detection_area.bearing_deg").as_double();
    model_config_.detection_area_fov_deg = get_parameter("detection_area.fov_deg").as_double();
  }
  validateModelConfigOrThrow();
}

void PointCloudObjectDetection::validateModelConfigOrThrow() const {
  auto fail = [this](const std::string& param, const std::string& message) {
    throwParameterError(this->get_logger(), param, message);
  };

  if (model_config_.intensity_threshold < 0) {
    fail("intensity_threshold", "must be non-negative");
  }

  if (!isFinite(model_config_.x_min) || !isFinite(model_config_.x_max)) {
    fail("x_min/x_max", "must be finite");
  }
  if (!(model_config_.x_min < model_config_.x_max)) {
    fail("x_min/x_max", "requires x_min < x_max");
  }

  if (!isFinite(model_config_.y_min) || !isFinite(model_config_.y_max)) {
    fail("y_min/y_max", "must be finite");
  }
  if (!(model_config_.y_min < model_config_.y_max)) {
    fail("y_min/y_max", "requires y_min < y_max");
  }

  if (!isFinite(model_config_.z_min) || !isFinite(model_config_.z_max)) {
    fail("z_min/z_max", "must be finite");
  }
  if (!(model_config_.z_min < model_config_.z_max)) {
    fail("z_min/z_max", "requires z_min < z_max");
  }

  if (model_config_.x_grid_size <= 0) {
    fail("x_grid_size", "must be greater than zero");
  }
  if (model_config_.y_grid_size <= 0) {
    fail("y_grid_size", "must be greater than zero");
  }

  if (model_config_.predicted_class_names.empty()) {
    fail("predicted_class_names", "must not be empty");
  }
  for (const auto& name : model_config_.predicted_class_names) {
    if (name.empty()) {
      fail("predicted_class_names", "class names must not be empty strings");
    }
  }

  if (model_type_ == ModelType::PBOD || model_type_ == ModelType::PP) {
    if (!isProbability(model_config_.nms_iou_threshold)) {
      fail("nms_iou_threshold", "must be within [0.0, 1.0]");
    }
    if (model_config_.nms_score_threshold.empty()) {
      fail("nms_score_threshold", "must not be empty");
    }
    for (double threshold : model_config_.nms_score_threshold) {
      if (!isProbability(threshold)) {
        fail("nms_score_threshold", "each entry must be within [0.0, 1.0]");
      }
    }
    if (model_config_.nms_max_num_objects < 0) {
      fail("nms_max_num_objects", "must be zero or positive");
    }
  }

  if (model_type_ == ModelType::PBOD || model_type_ == ModelType::TPOD) {
    if (model_config_.max_num_points <= 0) {
      fail("max_num_points", "must be greater than zero");
    }
    if (model_config_.num_point_features <= 0) {
      fail("num_point_features", "must be greater than zero");
    }
    PointType config_point_type = parsePointType(model_config_.point_type, this->get_logger());
    if (config_point_type == PointType::XYZI) {
      if (model_config_.num_point_features != 1) {
        fail("num_point_features", "must be 1 when point_type is PointXYZI");
      }
    } else if (config_point_type == PointType::XYZRV) {
      if (model_config_.num_point_features != 2) {
        fail("num_point_features", "must be 2 when point_type is PointXYZRV");
      }
    }
    if (model_config_.stride.empty()) {
      fail("stride", "must not be empty");
    }
    for (auto stride : model_config_.stride) {
      if (stride <= 0) {
        fail("stride", "entries must be positive integers");
      }
    }
    if (model_config_.first_up_stride <= 0) {
      fail("first_up_stride", "must be greater than zero");
    }

    if (model_config_.no_detection_zone_remove_points) {
      if (!(model_config_.no_detection_zone_x_min < model_config_.no_detection_zone_x_max) ||
          !(model_config_.no_detection_zone_y_min < model_config_.no_detection_zone_y_max)) {
        fail("no_detection_zone.remove_points", "requires x_min < x_max and y_min < y_max");
      }
    }
  }

  if (model_type_ == ModelType::TPOD) {
    if (!isProbability(model_config_.cls_threshold)) {
      fail("cls_threshold", "must be within [0.0, 1.0]");
    }
  }

  if (model_config_.detection_area_remove_points_outside) {
    if (model_config_.detection_area_radius <= 0.0) {
      fail("detection_area.radius", "must be greater than 0 when detection area filtering is enabled");
    }
    if (model_config_.detection_area_fov_deg <= 0.0 || model_config_.detection_area_fov_deg > 360.0) {
      fail("detection_area.fov_deg", "must be in the range (0, 360]");
    }
  }

  if (model_type_ == ModelType::PP) {
    if (model_config_.max_pillars <= 0) {
      fail("max_pillars", "must be greater than zero");
    }
    if (model_config_.max_points_per_pillar <= 0) {
      fail("max_points_per_pillar", "must be greater than zero");
    }
    if (model_config_.n_features <= 0) {
      fail("n_features", "must be greater than zero");
    }
    if (model_config_.downscaling <= 0) {
      fail("downscaling", "must be greater than zero");
    }
    if (!isFinite(model_config_.delta_x) || model_config_.delta_x <= 0.0) {
      fail("delta_x", "must be positive");
    }
    if (!isFinite(model_config_.delta_y) || model_config_.delta_y <= 0.0) {
      fail("delta_y", "must be positive");
    }
    if (model_config_.anchor_boxes.empty()) {
      fail("anchors", "at least one anchor must be configured");
    }
    if (model_config_.anchor_boxes.size() != model_config_.anchor_diagonals.size()) {
      fail("anchors", "internal anchor configuration is inconsistent");
    }
    for (std::size_t i = 0; i < model_config_.anchor_boxes.size(); ++i) {
      const auto& anchor = model_config_.anchor_boxes[i];
      if (!isFinite(anchor.length) || anchor.length <= 0.0) {
        fail("anchors.length", "must be positive");
      }
      if (!isFinite(anchor.width) || anchor.width <= 0.0) {
        fail("anchors.width", "must be positive");
      }
      if (!isFinite(anchor.height) || anchor.height <= 0.0) {
        fail("anchors.height", "must be positive");
      }
      if (!isFinite(anchor.z_center)) {
        fail("anchors.z_center", "must be finite");
      }
      if (!isFinite(anchor.yaw)) {
        fail("anchors.yaw", "must be finite");
      }
    }
  }
}

rcl_interfaces::msg::SetParametersResult PointCloudObjectDetection::parametersCallback(
    const std::vector<rclcpp::Parameter>& parameters) {
  rcl_interfaces::msg::SetParametersResult result;
  result.successful = true;
  result.reason = "success";
  bool model_change_on_runtime = false;
  bool publishers_changed = false;

  for (const auto& param : parameters) {
    // General parameters
    if (param.get_name() == "model_name") {
      std::string new_model_name = param.as_string();
      if (new_model_name != params_.model_name) {
        params_.model_name = new_model_name;
        model_change_on_runtime = true;
        RCLCPP_INFO(this->get_logger(), "Model name changed to: %s", params_.model_name.c_str());
      }
    } else if (param.get_name() == "model_version") {
      std::string new_model_version = param.as_string();
      if (new_model_version != params_.model_version) {
        params_.model_version = new_model_version;
        model_change_on_runtime = true;
        RCLCPP_INFO(this->get_logger(), "Model version changed to: %s", params_.model_version.c_str());
      }
    } else if (param.get_name() == "inference_frame") {
      params_.inference_frame = param.as_string();
    } else if (param.get_name() == "output_frame") {
      params_.output_frame = param.as_string();
    } else if (param.get_name() == "variances") {
      auto cscu = pm::object_access::CONTINUOUS_STATE_COVARIANCE_UNKNOWN;
      get_parameter_or("variances", params_.variance,
                       {cscu, cscu, cscu, cscu, cscu, cscu, cscu, cscu, cscu, cscu, cscu, cscu});
      sanitizeVarianceVector(params_.variance, cscu);
    } else if (param.get_name() == "class_score_threshold") {
      params_.class_score_threshold = param.as_double();
    } else if (param.get_name() == "predicted_class_names") {
      model_config_.predicted_class_names = param.as_string_array();
      model_change_on_runtime = true;
    } else if (param.get_name() == "intensity_threshold") {
      model_config_.intensity_threshold = param.as_int();
      model_change_on_runtime = true;
    } else if (param.get_name() == "with_velocity") {
      model_config_.with_velocity = param.as_bool();
      model_change_on_runtime = true;
    } else if (param.get_name() == "sensor_id") {
      const int64_t sensor_id_value = param.as_int();
      if (sensor_id_value < -1) {
        result.successful = false;
        result.reason = "sensor_id must be -1 (random) or a non-negative integer";
        RCLCPP_ERROR(this->get_logger(), "%s", result.reason.c_str());
        return result;
      }
      params_.sensor_id =
          sensor_id_value == -1 ? static_cast<uint64_t>(random()) : static_cast<uint64_t>(sensor_id_value);
    } else if (param.get_name() == "server_url") {
      std::string new_server_url = param.as_string();
      if (new_server_url != params_.server_url) {
        params_.server_url = new_server_url;
        model_change_on_runtime = true;
        RCLCPP_INFO(this->get_logger(), "Server URL changed to: %s", params_.server_url.c_str());
      }
    } else if (param.get_name() == "use_shm") {
      bool new_use_shm = param.as_bool();
      if (new_use_shm != params_.use_shm) {
        params_.use_shm = new_use_shm;
        model_change_on_runtime = true;
        RCLCPP_INFO(this->get_logger(), "Use shared memory changed to: %s", params_.use_shm ? "true" : "false");
      }
    }

    // NMS parameters
    else if (param.get_name() == "nms_max_num_objects") {
      model_config_.nms_max_num_objects = param.as_int();
      model_change_on_runtime = true;
    } else if (param.get_name() == "nms_iou_threshold") {
      model_config_.nms_iou_threshold = param.as_double();
      model_change_on_runtime = true;
    } else if (param.get_name() == "nms_score_threshold") {
      model_config_.nms_score_threshold = param.as_double_array();
      model_change_on_runtime = true;
    }

    // No-detection zone parameters
    else if (param.get_name() == "no_detection_zone.enabled") {
      params_.no_detection_zone_enabled = param.as_bool();
    } else if (param.get_name() == "no_detection_zone.x_min") {
      params_.no_detection_zone_x_min = param.as_double();
      model_config_.no_detection_zone_x_min = param.as_double();
    } else if (param.get_name() == "no_detection_zone.x_max") {
      params_.no_detection_zone_x_max = param.as_double();
      model_config_.no_detection_zone_x_max = param.as_double();
    } else if (param.get_name() == "no_detection_zone.y_min") {
      params_.no_detection_zone_y_min = param.as_double();
      model_config_.no_detection_zone_y_min = param.as_double();
    } else if (param.get_name() == "no_detection_zone.y_max") {
      params_.no_detection_zone_y_max = param.as_double();
      model_config_.no_detection_zone_y_max = param.as_double();
    } else if (param.get_name() == "no_detection_zone.publish_polygon") {
      params_.no_detection_zone_publish_polygon = param.as_bool();
      publishers_changed = true;
    } else if (param.get_name() == "detection_area.enabled") {
      params_.detection_area_enabled = param.as_bool();
    } else if (param.get_name() == "detection_area.center_x") {
      params_.detection_area_center_x = param.as_double();
      model_config_.detection_area_center_x = param.as_double();
    } else if (param.get_name() == "detection_area.center_y") {
      params_.detection_area_center_y = param.as_double();
      model_config_.detection_area_center_y = param.as_double();
    } else if (param.get_name() == "detection_area.radius") {
      params_.detection_area_radius = param.as_double();
      model_config_.detection_area_radius = param.as_double();
    } else if (param.get_name() == "detection_area.bearing_deg") {
      params_.detection_area_bearing_deg = param.as_double();
      model_config_.detection_area_bearing_deg = param.as_double();
    } else if (param.get_name() == "detection_area.fov_deg") {
      params_.detection_area_fov_deg = param.as_double();
      model_config_.detection_area_fov_deg = param.as_double();
    } else if (param.get_name() == "detection_area.num_segments") {
      params_.detection_area_num_segments = param.as_int();
    } else if (param.get_name() == "detection_area.filter_detections") {
      params_.detection_area_filter_detections = param.as_bool();
    } else if (param.get_name() == "detection_area.filter_mode") {
      params_.detection_area_filter_mode = param.as_string();
    } else if (param.get_name() == "detection_area.publish_polygon") {
      params_.detection_area_publish_polygon = param.as_bool();
      publishers_changed = true;
    }

    // Grid parameters
    else if (param.get_name() == "x_min") {
      model_config_.x_min = param.as_double();
      model_change_on_runtime = true;
    } else if (param.get_name() == "x_max") {
      model_config_.x_max = param.as_double();
      model_change_on_runtime = true;
    } else if (param.get_name() == "y_min") {
      model_config_.y_min = param.as_double();
      model_change_on_runtime = true;
    } else if (param.get_name() == "y_max") {
      model_config_.y_max = param.as_double();
      model_change_on_runtime = true;
    } else if (param.get_name() == "z_min") {
      model_config_.z_min = param.as_double();
      model_change_on_runtime = true;
    } else if (param.get_name() == "z_max") {
      model_config_.z_max = param.as_double();
      model_change_on_runtime = true;
    } else if (param.get_name() == "x_grid_size") {
      model_config_.x_grid_size = param.as_int();
      model_change_on_runtime = true;
    } else if (param.get_name() == "y_grid_size") {
      model_config_.y_grid_size = param.as_int();
      model_change_on_runtime = true;
    }

    // PBOD/TPOD-specific parameters
    else if (param.get_name() == "max_num_points") {
      model_config_.max_num_points = param.as_int();
      model_change_on_runtime = true;
    } else if (param.get_name() == "stride") {
      model_config_.stride = param.as_integer_array();
      model_change_on_runtime = true;
    } else if (param.get_name() == "first_up_stride") {
      model_config_.first_up_stride = param.as_int();
      model_change_on_runtime = true;
    } else if (param.get_name() == "mask_is_bool") {
      model_config_.mask_is_bool = param.as_bool();
      model_change_on_runtime = true;
    } else if (param.get_name() == "zero_intensity") {
      model_config_.zero_intensity = param.as_bool();
      model_change_on_runtime = true;
    } else if (param.get_name() == "no_detection_zone.remove_points") {
      model_config_.no_detection_zone_remove_points = param.as_bool();
    } else if (param.get_name() == "cls_threshold") {
      model_config_.cls_threshold = param.as_double();
      model_change_on_runtime = true;
    } else if (param.get_name() == "num_point_features") {
      model_config_.num_point_features = param.as_int();
      model_change_on_runtime = true;
    } else if (param.get_name() == "point_type") {
      model_config_.point_type = param.as_string();
      point_type_ = parsePointType(model_config_.point_type, this->get_logger());
      model_change_on_runtime = true;
    }

    // PP-specific parameters
    else if (param.get_name() == "max_pillars") {
      model_config_.max_pillars = param.as_int();
      model_change_on_runtime = true;
    } else if (param.get_name() == "max_points_per_pillar") {
      model_config_.max_points_per_pillar = param.as_int();
      model_change_on_runtime = true;
    } else if (param.get_name() == "n_features") {
      model_config_.n_features = param.as_int();
      model_change_on_runtime = true;
    } else if (param.get_name() == "downscaling") {
      model_config_.downscaling = param.as_int();
      model_change_on_runtime = true;
    } else if (param.get_name() == "anchors_string") {
      model_config_.anchors_string = param.as_string();
      model_change_on_runtime = true;
    }

    // Publishing control parameters
    else if (param.get_name() == "publish_class_point_clouds") {
      params_.publish_class_point_clouds = param.as_bool();
      publishers_changed = true;
    } else if (param.get_name() == "publish_unclassified_points") {
      params_.publish_unclassified_points = param.as_bool();
      publishers_changed = true;
    } else if (param.get_name() == "publish_unclassified_points_outside_detection_area") {
      params_.publish_unclassified_points_outside_detection_area = param.as_bool();
      publishers_changed = true;
    } else if (param.get_name() == "no_detection_zone.publish_points") {
      params_.no_detection_zone_publish_points = param.as_bool();
      publishers_changed = true;
    } else if (param.get_name() == "model_bounds.publish_polygon") {
      params_.model_bounds_publish_polygon = param.as_bool();
      publishers_changed = true;
    }
  }

  // Reinitialize model if model name or version changed
  if (model_change_on_runtime) {
    try {
      model_runtime_overwrite_ = true;
      initializeModel();
      RCLCPP_INFO(this->get_logger(), "Successfully reinitialized model with name: %s, version: %s",
                  params_.model_name.c_str(), params_.model_version.c_str());
    } catch (const std::exception& e) {
      result.successful = false;
      result.reason = "Failed to reinitialize model: " + std::string(e.what());
      RCLCPP_ERROR(this->get_logger(), "Failed to reinitialize model: %s", e.what());
      return result;
    }
  }

  validateParamsOrThrow();
  if (detection_model_) {
    validateModelConfigOrThrow();
  }

  // Update publishers only if publishing control parameters changed
  if (publishers_changed) {
    setupPublishers();
  }

  return result;
}

void PointCloudObjectDetection::initializeModel() {
  // load model
  using namespace std::string_literals;
  triton_interface_ = std::make_unique<triton_cpp::TritonInterface>(params_.model_name, params_.model_version,
                                                                    params_.server_url, params_.use_shm, false, true);

  // log model info
  std::cout << triton_interface_->getModelInfo() << std::endl;

  // create model architecture
  if (triton_interface_->nInputs() == 3 && triton_interface_->nOutputs() == 4) {
    model_type_ = ModelType::PBOD;
    loadModelConfig();
    detection_model_ = std::make_unique<PBODModel>(*triton_interface_.get(), model_config_);
    std::map<std::string, std::vector<int64_t>> o{{"reg_logits"s, {65536l, 7l}}};
  } else if (triton_interface_->nInputs() == 2 &&
             (triton_interface_->nOutputs() == 6 || triton_interface_->nOutputs() == 7)) {
    model_type_ = ModelType::PP;
    loadModelConfig();
    detection_model_ = std::make_unique<PPModel>(*triton_interface_.get(), model_config_, params_);
    // } else if (triton_interface_->nInputs() == 3 && triton_interface_->nOutputs() == 2) {
    //   model_type_ = ModelType::TPOD;
    //   loadModelConfig();
    //   detection_model_ = std::make_unique<TPODModel>(*triton_interface_.get(), model_config_);
  } else {
    RCLCPP_FATAL(this->get_logger(),
                 "Model error: The number of model inputs or outputs does not correspond with model "
                 "architecture of PP, PBOD, or TPOD.");
    exit(EXIT_FAILURE);
  }
  triton_interface_->initInOutputs(detection_model_->getSpecialOutputShapes());

  if (model_type_ != ModelType::TPOD) {
    // intialize params for non maximum suppression and object list creator
    non_max_suppression_ = std::make_unique<point_cloud_object_detection::NonMaxSuppression>(model_config_, params_);
  }
}

void PointCloudObjectDetection::setupPublishers() {
  point_cloud_transport::PointCloudTransport pct(this->shared_from_this());

  // create no-detection zone polygon publisher if enabled
  if (params_.no_detection_zone_publish_polygon) {
    if (!no_detection_zone_pub_) {
      no_detection_zone_pub_ = create_publisher<geometry_msgs::msg::PolygonStamped>(kNoDetectionZoneTopic, 1);
      RCLCPP_INFO(this->get_logger(), "Publishing no-detection zone polygon on '%s'",
                  no_detection_zone_pub_->get_topic_name());
    }
  } else {
    no_detection_zone_pub_.reset();
  }

  // create detection area polygon publisher if requested
  if (params_.detection_area_publish_polygon) {
    if (!detection_area_pub_) {
      detection_area_pub_ = create_publisher<geometry_msgs::msg::PolygonStamped>(kDetectionAreaTopic, 1);
      RCLCPP_INFO(this->get_logger(), "Publishing detection area polygon on '%s'",
                  detection_area_pub_->get_topic_name());
    }
  } else {
    detection_area_pub_.reset();
  }

  // create model bounds polygon publisher if requested
  if (params_.model_bounds_publish_polygon) {
    if (!model_bounds_pub_) {
      model_bounds_pub_ = create_publisher<geometry_msgs::msg::PolygonStamped>(kModelBoundsTopic, 1);
      RCLCPP_INFO(this->get_logger(), "Publishing model bounds polygon on '%s'", model_bounds_pub_->get_topic_name());
    }
  } else {
    model_bounds_pub_.reset();
  }

  // create class-specific point cloud publishers if enabled
  if (params_.publish_class_point_clouds) {
    if (class_publishers_.empty()) {
      for (const auto& class_name : model_config_.predicted_class_names) {
        std::string topic_name = "~/class_point_clouds/" + sanitizeTopicName(class_name);
        topic_name = this->get_node_topics_interface()->resolve_topic_name(topic_name);
        class_publishers_[class_name] =
            std::make_shared<point_cloud_transport::Publisher>(pct.advertise(topic_name, 1));
        RCLCPP_INFO(this->get_logger(), "Publishing class '%s' points to '%s'", class_name.c_str(),
                    class_publishers_[class_name]->getTopic().c_str());
      }
    }
  } else {
    class_publishers_.clear();
  }

  // create unclassified points publisher if enabled
  if (params_.publish_unclassified_points) {
    if (!unclassified_publisher_) {
      std::string topic_name = this->get_node_topics_interface()->resolve_topic_name(kUnclassifiedPointsTopic);
      unclassified_publisher_ = std::make_shared<point_cloud_transport::Publisher>(pct.advertise(topic_name, 1));
      RCLCPP_INFO(this->get_logger(), "Publishing unclassified points to '%s'",
                  unclassified_publisher_->getTopic().c_str());
    }
  } else {
    unclassified_publisher_.reset();
  }

  // create unclassified points outside detection area publisher if enabled
  if (params_.publish_unclassified_points_outside_detection_area) {
    if (!unclassified_outside_area_publisher_) {
      std::string topic_name = this->get_node_topics_interface()->resolve_topic_name(kUnclassifiedOutsideAreaTopic);
      unclassified_outside_area_publisher_ =
          std::make_shared<point_cloud_transport::Publisher>(pct.advertise(topic_name, 1));
      RCLCPP_INFO(this->get_logger(), "Publishing unclassified points outside detection area to '%s'",
                  unclassified_outside_area_publisher_->getTopic().c_str());
    }
  } else {
    unclassified_outside_area_publisher_.reset();
  }

  // create no-detection zone raw points publisher if enabled
  if (params_.no_detection_zone_publish_points) {
    if (!no_detection_zone_points_publisher_) {
      std::string topic_name = this->get_node_topics_interface()->resolve_topic_name(kNoDetectionZonePointsTopic);
      no_detection_zone_points_publisher_ =
          std::make_shared<point_cloud_transport::Publisher>(pct.advertise(topic_name, 1));
      RCLCPP_INFO(this->get_logger(), "Publishing no-detection zone points to '%s'",
                  no_detection_zone_points_publisher_->getTopic().c_str());
    }
  } else {
    no_detection_zone_points_publisher_.reset();
  }
}

void PointCloudObjectDetection::setup() {
  // initially load model
  initializeModel();

  // create a callback for dynamic parameter configuration
  parameters_callback_ = this->add_on_set_parameters_callback(
      std::bind(&PointCloudObjectDetection::parametersCallback, this, std::placeholders::_1));

  // create a transform buffer and listener
  tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
  tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

  // create subscriber and publisher
  std::string resolved_input_topic = this->get_node_topics_interface()->resolve_topic_name(kInputTopic);
  point_cloud_transport::PointCloudTransport pct(this->shared_from_this());
  subscriber_ = std::make_shared<point_cloud_transport::Subscriber>(pct.subscribe(
      resolved_input_topic, 1, std::bind(&PointCloudObjectDetection::predict, this, std::placeholders::_1),
      std::shared_ptr<void>()));
  publisher_ = create_publisher<pm::msg::ObjectList>(kOutputTopic, 1);
  RCLCPP_INFO(this->get_logger(), "Subscribed to '%s'", subscriber_->getTopic().c_str());
  RCLCPP_INFO(this->get_logger(), "Publishing to '%s'", publisher_->get_topic_name());

  // Setup all publishers
  setupPublishers();
}

void PointCloudObjectDetection::processPointCloud(const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg,
                                                  PointCloud& point_cloud) {
  // transform sensor_msgs::msg::PointCloud2 msg if required
  sensor_msgs::msg::PointCloud2::SharedPtr transformed_point_cloud_msg;

  if (!params_.inference_frame.empty() && msg->header.frame_id != params_.inference_frame) {
    transformed_point_cloud_msg = std::make_shared<sensor_msgs::msg::PointCloud2>();
    // transform point cloud
    try {
      tf_buffer_->transform(*msg, *transformed_point_cloud_msg, params_.inference_frame);
    } catch (tf2::TransformException& e) {
      RCLCPP_ERROR(this->get_logger(),
                   "Cannot tranform Pointcloud: Transformation from its frame (%s) to inference_frame "
                   "(%s) not found: %s",
                   msg->header.frame_id.c_str(), params_.inference_frame.c_str(), e.what());
      return;
    }
  } else {
    transformed_point_cloud_msg = std::make_shared<sensor_msgs::msg::PointCloud2>(*msg);
  }

  // Always convert base XYZ + intensity
  pcl::PointCloud<pcl::PointXYZI> base_cloud;
  pcl::fromROSMsg(*transformed_point_cloud_msg, base_cloud);

  point_cloud.clear();
  point_cloud.header = base_cloud.header;
  point_cloud.width = base_cloud.width;
  point_cloud.height = base_cloud.height;
  point_cloud.is_dense = base_cloud.is_dense;
  point_cloud.points.resize(base_cloud.size());

  auto findField = [&](const std::string& name) -> const sensor_msgs::msg::PointField* {
    for (const auto& field : transformed_point_cloud_msg->fields) {
      if (field.name == name) {
        return &field;
      }
    }
    return nullptr;
  };

  const bool use_velocity_channel = (point_type_ == PointType::XYZRV);
  const std::size_t extra_channels = use_velocity_channel ? 1U : 0U;
  if (extra_channels == 0) {
    extra_feature_buffer_.clear();
  } else {
    extra_feature_buffer_.assign(point_cloud.points.size() * extra_channels, 0.0f);
  }

  std::unique_ptr<sensor_msgs::PointCloud2ConstIterator<float>> reflectivity_iter;
  std::unique_ptr<sensor_msgs::PointCloud2ConstIterator<float>> velocity_iter;
  static bool reflectivity_warned = false;
  static bool velocity_warned = false;

  if (use_velocity_channel) {
    const auto* reflectivity_field = findField("reflectivity");
    if (reflectivity_field && reflectivity_field->datatype == sensor_msgs::msg::PointField::FLOAT32) {
      reflectivity_iter =
          std::make_unique<sensor_msgs::PointCloud2ConstIterator<float>>(*transformed_point_cloud_msg, "reflectivity");
    } else if (!reflectivity_warned) {
      RCLCPP_WARN(this->get_logger(), "Point field 'reflectivity' missing or not FLOAT32; falling back to 'intensity'");
      reflectivity_warned = true;
    }

    const auto* velocity_field = findField("velocity");
    if (velocity_field && velocity_field->datatype == sensor_msgs::msg::PointField::FLOAT32) {
      velocity_iter =
          std::make_unique<sensor_msgs::PointCloud2ConstIterator<float>>(*transformed_point_cloud_msg, "velocity");
    } else if (!velocity_warned) {
      RCLCPP_WARN(this->get_logger(), "Point field 'velocity' missing or not FLOAT32; velocity channel will be zero");
      velocity_warned = true;
    }
  }

  for (std::size_t idx = 0; idx < base_cloud.size(); ++idx) {
    const auto& src = base_cloud.points[idx];
    auto& dst = point_cloud.points[idx];
    dst.x = src.x;
    dst.y = src.y;
    dst.z = src.z;
    float intensity_value = src.intensity;
    if (reflectivity_iter) {
      intensity_value = **reflectivity_iter;
      ++(*reflectivity_iter);
    }

    dst.intensity = intensity_value;

    if (use_velocity_channel) {
      float velocity_value = 0.0f;
      if (velocity_iter) {
        velocity_value = **velocity_iter;
        ++(*velocity_iter);
      }
      extra_feature_buffer_[idx * extra_channels] = velocity_value;
    }
  }
}

void PointCloudObjectDetection::boxesToObjectList(const std::vector<BoundingBox>& bboxes,
                                                  pm::msg::ObjectList& object_list) {
  // iterate over all boxes
  for (int idx = 0; idx < int(bboxes.size()); ++idx) {
    pm::msg::Object object;
    pm::object_access::initializeState(object, pm::msg::ISCACTR::MODEL_ID);

    // Set sensor ID
    object.state.sensor_id.push_back(params_.sensor_id);

    // set id
    object.id = idx;

    // set occupancy probability
    object.existence_probability = bboxes[idx].existence_probability;

    // set object position
    pm::object_access::setX(object, bboxes[idx].center[0]);
    pm::object_access::setY(object, bboxes[idx].center[1]);
    pm::object_access::setZ(object, bboxes[idx].z);
    pm::object_access::setYaw(object, bboxes[idx].yaw);

    // set object dimensions
    pm::object_access::setLength(object, bboxes[idx].length);
    pm::object_access::setWidth(object, bboxes[idx].width);
    pm::object_access::setHeight(object, bboxes[idx].height);

    // set velocity
    if (bboxes[idx].has_velocity) {
      geometry_msgs::msg::Vector3 velocity;
      velocity.x = bboxes[idx].v_x;
      velocity.y = bboxes[idx].v_y;
      velocity.z = 0.0;
      pm::object_access::setVelocityXYZYaw(object, velocity, bboxes[idx].yaw, false);
    }

    // set variance
    std::vector<double> variance(params_.variance.begin(), params_.variance.end());
    pm::object_access::setContinuousStateCovarianceDiagonal(object, variance);

    // get class probability
    for (auto& classification_entry : bboxes[idx].classification) {
      pm::msg::ObjectClassification cls;
      cls.probability = classification_entry.score;
      const std::string& cls_name = model_config_.predicted_class_names[classification_entry.class_idx];
      cls.type = model_config_.class_mapping_[cls_name];
      if (auto it = std::find_if(object.state.classifications.begin(), object.state.classifications.end(),
                                 [&cls](const pm::msg::ObjectClassification& c) { return c.type == cls.type; });
          it != object.state.classifications.end()) {
        // If this class already exists, add the probability to the existing one
        if (model_type_ == ModelType::PBOD) {
          // Note that PBOD at this point stores logits, so we have to add them with logaddexp
          it->probability = logaddexp(it->probability, cls.probability);
        } else {
          it->probability += cls.probability;
        }

      } else {
        // Else, add this class
        object.state.classifications.push_back(cls);
      }
    }
    // normalize class probabilities and remove unlikely classes
    sanitize_classifications(object.state.classifications, params_.class_score_threshold,
                             model_type_ == ModelType::PBOD);
    // add to object list
    object_list.objects.push_back(object);
  }

  // transform if required
  if (!params_.output_frame.empty() && params_.inference_frame != params_.output_frame) {
    geometry_msgs::msg::TransformStamped t;
    try {
      t = tf_buffer_->lookupTransform(params_.output_frame, params_.inference_frame, object_list.header.stamp);
    } catch (tf2::TransformException& e) {
      RCLCPP_ERROR(this->get_logger(),
                   "Cannot transform Object list: Transformation from inference_frame (%s) to output_frame "
                   "(%s) not found: %s",
                   params_.inference_frame.c_str(), params_.output_frame.c_str(), e.what());
      return;
    }
    pm::msg::ObjectList object_list_trans;
    tf2::doTransform(object_list, object_list_trans, t);
    object_list = object_list_trans;
    object_list.header.frame_id = params_.output_frame;
  } else if (!params_.output_frame.empty()) {
    // if inference = output frame, just change frame in header, but no transform needed
    object_list.header.frame_id = params_.output_frame;
  } else if (!params_.inference_frame.empty()) {
    // if output frame is empty and inference frame is not empty, just change frame in header, but no transform needed
    object_list.header.frame_id = params_.inference_frame;
  }
}

bool PointCloudObjectDetection::isPointInsideBoundingBox(const Point& point, const BoundingBox& bbox) {
  // Transform point to bounding box local coordinate system
  float cos_yaw = std::cos(bbox.yaw);
  float sin_yaw = std::sin(bbox.yaw);

  // Translate point to bbox center
  float dx = point.x - bbox.center[0];
  float dy = point.y - bbox.center[1];
  float dz = point.z - bbox.z;

  // Rotate point to align with bounding box axes
  float local_x = dx * cos_yaw + dy * sin_yaw;
  float local_y = -dx * sin_yaw + dy * cos_yaw;
  float local_z = dz;

  // Check if point is within bounding box dimensions
  return (std::abs(local_x) <= bbox.length / 2.0f) && (std::abs(local_y) <= bbox.width / 2.0f) &&
         (std::abs(local_z) <= bbox.height / 2.0f);
}

void PointCloudObjectDetection::publishClassPointClouds(const std::vector<BoundingBox>& bboxes,
                                                        const std_msgs::msg::Header& header,
                                                        const PointCloud& transformed_input_cloud) {
  if ((!params_.publish_class_point_clouds || class_publishers_.empty()) &&
      (!params_.publish_unclassified_points || !unclassified_publisher_)) {
    return;
  }

  // Get the filtered input points that were actually used by the model
  const PointCloud& filtered_points = detection_model_->getFilteredInputPoints();
  if (filtered_points.empty()) {
    RCLCPP_WARN(this->get_logger(), "No filtered points available for class-specific point cloud publishing");
  }
  // Create map to store points for each class
  std::map<std::string, PointCloud> class_point_clouds;
  PointCloud unclassified_points;
  PointCloud unclassified_points_outside_area;

  // Determine the correct frame_id for the point clouds
  // The filtered points are in the inference frame (if set) or original frame
  std::string point_cloud_frame_id = params_.inference_frame.empty() ? header.frame_id : params_.inference_frame;

  // Initialize point clouds for each class (even if empty)
  if (params_.publish_class_point_clouds) {
    for (const auto& class_name : model_config_.predicted_class_names) {
      class_point_clouds[class_name] = PointCloud();
      class_point_clouds[class_name].header.frame_id = point_cloud_frame_id;
      pcl_conversions::toPCL(header.stamp, class_point_clouds[class_name].header.stamp);
    }
  }

  // Initialize unclassified points cloud
  if (params_.publish_unclassified_points) {
    unclassified_points.header.frame_id = point_cloud_frame_id;
    pcl_conversions::toPCL(header.stamp, unclassified_points.header.stamp);
  }
  if (params_.publish_unclassified_points_outside_detection_area) {
    unclassified_points_outside_area.header.frame_id = point_cloud_frame_id;
    pcl_conversions::toPCL(header.stamp, unclassified_points_outside_area.header.stamp);
  }

  // Helpers to apply effective detection area constraints to published points
  auto is_inside_sector = [&](double x, double y) -> bool {
    if (!params_.detection_area_enabled) return true;
    const double cx = params_.detection_area_center_x;
    const double cy = params_.detection_area_center_y;
    const double r = params_.detection_area_radius;
    const double bearing = params_.detection_area_bearing_deg * M_PI / 180.0;
    const double fov = params_.detection_area_fov_deg * M_PI / 180.0;
    const double dx = x - cx;
    const double dy = y - cy;
    if (dx * dx + dy * dy > r * r) return false;
    double ang = std::atan2(dy, dx) - bearing;
    while (ang > M_PI) ang -= 2.0 * M_PI;
    while (ang < -M_PI) ang += 2.0 * M_PI;
    return std::abs(ang) <= (fov * 0.5 + 1e-9);
  };
  auto is_outside_no_detection = [&](double x, double y) -> bool {
    if (!params_.no_detection_zone_enabled) return true;
    // Exclude points in the active no-detection rectangle regardless of model input filtering
    return !(x >= params_.no_detection_zone_x_min && x <= params_.no_detection_zone_x_max &&
             y >= params_.no_detection_zone_y_min && y <= params_.no_detection_zone_y_max);
  };
  auto is_inside_model_bounds = [&](double x, double y) -> bool {
    // Model bounds (inference frame). These are always enforced by model input filtering,
    // but we keep this for consistency if models differ.
    return (x >= model_config_.x_min && x < model_config_.x_max && y >= model_config_.y_min && y < model_config_.y_max);
  };

  // If we have no points or no bboxes, publish empty clouds and return
  if (filtered_points.empty() || bboxes.empty()) {
    // Always publish empty class clouds for RViz clearing
    if (params_.publish_class_point_clouds) {
      for (const auto& [class_name, class_cloud] : class_point_clouds) {
        auto publisher_it = class_publishers_.find(class_name);
        if (publisher_it != class_publishers_.end()) {
          sensor_msgs::msg::PointCloud2 ros_msg;
          pcl::toROSMsg(class_cloud, ros_msg);
          ros_msg.header.stamp = header.stamp;
          ros_msg.header.frame_id = point_cloud_frame_id;
          publisher_it->second->publish(ros_msg);
          RCLCPP_DEBUG(this->get_logger(), "Cleared class '%s' point cloud (no detections)", class_name.c_str());
        }
      }
    }

    // If there are points but no detections, split unclassified by inside/outside effective area
    if ((params_.publish_unclassified_points && unclassified_publisher_) ||
        (params_.publish_unclassified_points_outside_detection_area && unclassified_outside_area_publisher_)) {
      unclassified_points.clear();
      unclassified_points_outside_area.clear();
      for (const auto& p : transformed_input_cloud) {
        const bool outside_ndz = is_outside_no_detection(p.x, p.y);
        const bool inside_bounds = is_inside_model_bounds(p.x, p.y);
        if (!outside_ndz || !inside_bounds) continue;
        if (is_inside_sector(p.x, p.y)) {
          if (params_.publish_unclassified_points) unclassified_points.push_back(p);
        } else {
          if (params_.publish_unclassified_points_outside_detection_area) {
            unclassified_points_outside_area.push_back(p);
          }
        }
      }

      if (params_.publish_unclassified_points && unclassified_publisher_) {
        sensor_msgs::msg::PointCloud2 ros_msg;
        pcl::toROSMsg(unclassified_points, ros_msg);
        ros_msg.header.stamp = header.stamp;
        ros_msg.header.frame_id = point_cloud_frame_id;
        unclassified_publisher_->publish(ros_msg);
        if (!unclassified_points.empty()) {
          RCLCPP_DEBUG(this->get_logger(), "Published all %zu points as unclassified (no detections)",
                       unclassified_points.size());
        } else {
          RCLCPP_DEBUG(this->get_logger(), "Unclassified point cloud empty (no points in original cloud)");
        }
      }
      if (params_.publish_unclassified_points_outside_detection_area && unclassified_outside_area_publisher_) {
        sensor_msgs::msg::PointCloud2 ros_msg2;
        pcl::toROSMsg(unclassified_points_outside_area, ros_msg2);
        ros_msg2.header.stamp = header.stamp;
        ros_msg2.header.frame_id = point_cloud_frame_id;
        unclassified_outside_area_publisher_->publish(ros_msg2);
      }
    }
    return;
  }

  // Convert point cloud to Eigen matrix for vectorized operations
  const size_t num_points = filtered_points.size();
  const size_t num_bboxes = bboxes.size();

  // Create matrices for all points (N x 3)
  Eigen::MatrixXf points_matrix(num_points, 3);
  for (size_t i = 0; i < num_points; ++i) {
    points_matrix(i, 0) = filtered_points[i].x;
    points_matrix(i, 1) = filtered_points[i].y;
    points_matrix(i, 2) = filtered_points[i].z;
  }

  // Track which points have been classified and to which class
  std::vector<bool> point_classified(num_points, false);
  std::vector<size_t> point_class_indices(num_points, SIZE_MAX);  // SIZE_MAX indicates unclassified

  // Pre-compute transformation data for all bounding boxes
  struct BBoxTransform {
    Eigen::Vector3f center;
    Eigen::Matrix2f rotation_inv;  // Inverse rotation matrix for 2D rotation
    Eigen::Vector3f half_extents;
    std::string best_class_name;
    size_t best_class_idx;
  };

  std::vector<BBoxTransform> bbox_transforms;
  bbox_transforms.reserve(num_bboxes);

  for (const auto& bbox : bboxes) {
    BBoxTransform transform;
    transform.center = Eigen::Vector3f(bbox.center[0], bbox.center[1], bbox.z);

    // Pre-compute rotation matrix (inverse rotation for transforming points to bbox local frame)
    float cos_yaw = std::cos(bbox.yaw);
    float sin_yaw = std::sin(bbox.yaw);
    transform.rotation_inv << cos_yaw, sin_yaw, -sin_yaw, cos_yaw;

    transform.half_extents = Eigen::Vector3f(bbox.length / 2.0f, bbox.width / 2.0f, bbox.height / 2.0f);

    // Find the best class for this bounding box
    // This logic determines which class-specific topic to publish points to based on the
    // highest-confidence classification result for this bounding box
    if (params_.publish_class_point_clouds && !bbox.classification.empty()) {
      // Search through all classification entries for this bounding box to find the one
      // with the highest confidence score. Each ClassificationEntry contains:
      // - class_idx: Index into the model's class name array
      // - score: Confidence score (0.0 to 1.0) for this classification
      auto max_class = std::max_element(
          bbox.classification.begin(), bbox.classification.end(),
          [](const ClassificationEntry& a, const ClassificationEntry& b) { return a.score < b.score; });

      // Validate that the class index is within bounds of our known class names
      // This prevents crashes if the model returns invalid class indices
      if (max_class->class_idx < model_config_.predicted_class_names.size()) {
        // Store the human-readable class name (e.g., "car", "pedestrian", "bus")
        // This will be used to determine which ROS topic to publish points to
        transform.best_class_name = model_config_.predicted_class_names[max_class->class_idx];
        transform.best_class_idx = max_class->class_idx;
      } else {
        // Invalid class index - treat as unclassified
        // Points in this bounding box will go to the unclassified topic if enabled
        transform.best_class_name = "";
        transform.best_class_idx = SIZE_MAX;
      }
    } else {
      // Either class publishing is disabled or this bounding box has no classification data
      // Mark as unclassified so points will go to the unclassified topic if enabled
      transform.best_class_name = "";
      transform.best_class_idx = SIZE_MAX;
    }

    bbox_transforms.push_back(std::move(transform));
  }

  // Vectorized point-in-bbox checking
  // Process points in batches for better cache performance
  constexpr size_t BATCH_SIZE = 1024;

  for (size_t batch_start = 0; batch_start < num_points; batch_start += BATCH_SIZE) {
    size_t batch_end = std::min(batch_start + BATCH_SIZE, num_points);
    size_t batch_size = batch_end - batch_start;

    // Extract batch of points
    Eigen::MatrixXf batch_points = points_matrix.block(batch_start, 0, batch_size, 3);

    // Check each bounding box against the current batch
    for (size_t bbox_idx = 0; bbox_idx < num_bboxes; ++bbox_idx) {
      const auto& transform = bbox_transforms[bbox_idx];

      // Translate points to bbox center
      Eigen::MatrixXf translated_points = batch_points.rowwise() - transform.center.transpose();

      // Rotate XY coordinates to bbox local frame
      Eigen::MatrixXf rotated_xy = translated_points.leftCols<2>() * transform.rotation_inv.transpose();

      // Check if points are inside the bounding box
      for (size_t i = 0; i < batch_size; ++i) {
        size_t global_idx = batch_start + i;

        // Skip if point is already classified
        if (point_classified[global_idx]) {
          continue;
        }

        // Check if point is within bounding box extents
        if (std::abs(rotated_xy(i, 0)) <= transform.half_extents.x() &&
            std::abs(rotated_xy(i, 1)) <= transform.half_extents.y() &&
            std::abs(translated_points(i, 2)) <= transform.half_extents.z()) {
          point_classified[global_idx] = true;

          // Assign class if we're publishing class-specific point clouds
          if (params_.publish_class_point_clouds && transform.best_class_idx != SIZE_MAX) {
            point_class_indices[global_idx] = transform.best_class_idx;
          }
        }
      }
    }
  }

  // Assign points to their respective class point clouds or unclassified cloud
  for (size_t i = 0; i < num_points; ++i) {
    const auto& point = filtered_points[i];

    if (point_classified[i] && params_.publish_class_point_clouds && point_class_indices[i] != SIZE_MAX) {
      // Add to class-specific point cloud
      const std::string& class_name = model_config_.predicted_class_names[point_class_indices[i]];
      class_point_clouds[class_name].push_back(point);
    } else if (!point_classified[i]) {
      // Split into inside/outside-area unclassified topics. Always exclude NDZ and out-of-bounds.
      const bool outside_ndz = is_outside_no_detection(point.x, point.y);
      const bool inside_bounds = is_inside_model_bounds(point.x, point.y);
      if (!outside_ndz || !inside_bounds) {
        continue;
      }
      if (params_.publish_unclassified_points && is_inside_sector(point.x, point.y)) {
        unclassified_points.push_back(point);
      }
    }
  }

  // Add outside-sector unclassified points from the full transformed input cloud
  if (params_.publish_unclassified_points_outside_detection_area) {
    for (const auto& p : transformed_input_cloud) {
      if (is_inside_sector(p.x, p.y)) continue;  // only outside-sector points
      const bool outside_ndz = is_outside_no_detection(p.x, p.y);
      const bool inside_bounds = is_inside_model_bounds(p.x, p.y);
      if (!outside_ndz || !inside_bounds) continue;
      unclassified_points_outside_area.push_back(p);
    }
  }

  // Publish point clouds for each class - ALWAYS publish to ensure RViz clearing
  if (params_.publish_class_point_clouds) {
    for (const auto& [class_name, class_cloud] : class_point_clouds) {
      auto publisher_it = class_publishers_.find(class_name);
      if (publisher_it != class_publishers_.end()) {
        sensor_msgs::msg::PointCloud2 ros_msg;
        pcl::toROSMsg(class_cloud, ros_msg);
        // Keep the frame_id from the PCL cloud, but use the timestamp from the original header
        ros_msg.header.stamp = header.stamp;
        ros_msg.header.frame_id = point_cloud_frame_id;

        // Always publish, even if empty - this ensures RViz clears old points
        publisher_it->second->publish(ros_msg);

        if (!class_cloud.empty()) {
          RCLCPP_DEBUG(this->get_logger(), "Published %zu points for class '%s'", class_cloud.size(),
                       class_name.c_str());
        } else {
          RCLCPP_DEBUG(this->get_logger(), "Published empty point cloud for class '%s' (clearing RViz)",
                       class_name.c_str());
        }
      }
    }
  }

  // Publish unclassified points - always publish if enabled to ensure RViz clearing
  if (params_.publish_unclassified_points && unclassified_publisher_) {
    sensor_msgs::msg::PointCloud2 ros_msg;
    pcl::toROSMsg(unclassified_points, ros_msg);
    // Keep the frame_id from the PCL cloud, but use the timestamp from the original header
    ros_msg.header.stamp = header.stamp;
    ros_msg.header.frame_id = point_cloud_frame_id;

    // Always publish unclassified points (empty if all points were classified)
    unclassified_publisher_->publish(ros_msg);

    if (!unclassified_points.empty()) {
      RCLCPP_DEBUG(this->get_logger(), "Published %zu unclassified points", unclassified_points.size());
    } else {
      RCLCPP_DEBUG(this->get_logger(), "Published empty unclassified point cloud (clearing RViz)");
    }
  }

  // Publish unclassified points outside detection area if enabled (always publish to clear)
  if (params_.publish_unclassified_points_outside_detection_area && unclassified_outside_area_publisher_) {
    sensor_msgs::msg::PointCloud2 ros_msg2;
    pcl::toROSMsg(unclassified_points_outside_area, ros_msg2);
    ros_msg2.header.stamp = header.stamp;
    ros_msg2.header.frame_id = point_cloud_frame_id;
    unclassified_outside_area_publisher_->publish(ros_msg2);
    if (!unclassified_points_outside_area.empty()) {
      RCLCPP_DEBUG(this->get_logger(), "Published %zu unclassified points outside detection area",
                   unclassified_points_outside_area.size());
    } else {
      RCLCPP_DEBUG(this->get_logger(), "Published empty outside-area unclassified point cloud (clearing RViz)");
    }
  }
}

void PointCloudObjectDetection::predict(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& msg) {
  // initialize timer
  std::vector<std::chrono::time_point<std::chrono::high_resolution_clock>> timestamps = {
      std::chrono::high_resolution_clock::now()};  // index: 0, start timer

  // msg to transformed pointcloud
  PointCloud point_cloud;
  auto header = msg->header;
  processPointCloud(msg, point_cloud);
  timestamps.push_back(std::chrono::high_resolution_clock::now());  // index: 1, after pcl preprocessing

  const std::size_t extra_channels =
      model_config_.num_point_features > 1 ? static_cast<std::size_t>(model_config_.num_point_features - 1) : 0;
  const float* extra_features =
      (extra_channels > 0 && extra_feature_buffer_.size() == point_cloud.size() * extra_channels)
          ? extra_feature_buffer_.data()
          : nullptr;
  if (detection_model_) {
    detection_model_->setAdditionalPointFeatures(extra_features, extra_channels > 0 ? point_cloud.size() : 0,
                                                 extra_channels);
  }

  // Optionally publish raw points inside the no-detection zone
  if (params_.no_detection_zone_publish_points && params_.no_detection_zone_enabled &&
      no_detection_zone_points_publisher_) {
    PointCloud ndz_points;
    ndz_points.header.frame_id = params_.inference_frame.empty() ? msg->header.frame_id : params_.inference_frame;
    pcl_conversions::toPCL(msg->header.stamp, ndz_points.header.stamp);
    const double x_min = params_.no_detection_zone_x_min;
    const double x_max = params_.no_detection_zone_x_max;
    const double y_min = params_.no_detection_zone_y_min;
    const double y_max = params_.no_detection_zone_y_max;
    for (const auto& p : point_cloud) {
      if (p.x >= x_min && p.x <= x_max && p.y >= y_min && p.y <= y_max) {
        ndz_points.push_back(p);
      }
    }
    sensor_msgs::msg::PointCloud2 ros_msg;
    pcl::toROSMsg(ndz_points, ros_msg);
    ros_msg.header.stamp = msg->header.stamp;
    ros_msg.header.frame_id = ndz_points.header.frame_id;
    no_detection_zone_points_publisher_->publish(ros_msg);
  }

  // Publish the no-detection zone polygon if requested and enabled
  if (params_.no_detection_zone_publish_polygon && params_.no_detection_zone_enabled && no_detection_zone_pub_) {
    geometry_msgs::msg::PolygonStamped polygon_message;
    polygon_message.header.stamp = msg->header.stamp;
    // Use inference_frame if set; otherwise use point cloud frame
    polygon_message.header.frame_id = params_.inference_frame.empty() ? msg->header.frame_id : params_.inference_frame;

    geometry_msgs::msg::Point32 polygon_point;
    // (x_min, y_min)
    polygon_point.x = static_cast<float>(params_.no_detection_zone_x_min);
    polygon_point.y = static_cast<float>(params_.no_detection_zone_y_min);
    polygon_point.z = 0.0f;
    polygon_message.polygon.points.push_back(polygon_point);
    // (x_max, y_min)
    polygon_point.x = static_cast<float>(params_.no_detection_zone_x_max);
    polygon_point.y = static_cast<float>(params_.no_detection_zone_y_min);
    polygon_message.polygon.points.push_back(polygon_point);
    // (x_max, y_max)
    polygon_point.x = static_cast<float>(params_.no_detection_zone_x_max);
    polygon_point.y = static_cast<float>(params_.no_detection_zone_y_max);
    polygon_message.polygon.points.push_back(polygon_point);
    // (x_min, y_max)
    polygon_point.x = static_cast<float>(params_.no_detection_zone_x_min);
    polygon_point.y = static_cast<float>(params_.no_detection_zone_y_max);
    polygon_message.polygon.points.push_back(polygon_point);

    no_detection_zone_pub_->publish(polygon_message);
  }

  // Publish the model bounds rectangle as polygon if requested
  if (params_.model_bounds_publish_polygon && model_bounds_pub_) {
    geometry_msgs::msg::PolygonStamped polygon_message;
    polygon_message.header.stamp = msg->header.stamp;
    polygon_message.header.frame_id = params_.inference_frame.empty() ? msg->header.frame_id : params_.inference_frame;

    geometry_msgs::msg::Point32 p;
    // (x_min, y_min)
    p.x = static_cast<float>(model_config_.x_min);
    p.y = static_cast<float>(model_config_.y_min);
    p.z = 0.0f;
    polygon_message.polygon.points.push_back(p);
    // (x_max, y_min)
    p.x = static_cast<float>(model_config_.x_max);
    p.y = static_cast<float>(model_config_.y_min);
    polygon_message.polygon.points.push_back(p);
    // (x_max, y_max)
    p.x = static_cast<float>(model_config_.x_max);
    p.y = static_cast<float>(model_config_.y_max);
    polygon_message.polygon.points.push_back(p);
    // (x_min, y_max)
    p.x = static_cast<float>(model_config_.x_min);
    p.y = static_cast<float>(model_config_.y_max);
    polygon_message.polygon.points.push_back(p);

    model_bounds_pub_->publish(polygon_message);
  }

  // Publish the detection area sector polygon if requested and enabled
  if (params_.detection_area_publish_polygon && params_.detection_area_enabled && detection_area_pub_) {
    geometry_msgs::msg::PolygonStamped polygon_message;
    polygon_message.header.stamp = msg->header.stamp;
    polygon_message.header.frame_id = params_.inference_frame.empty() ? msg->header.frame_id : params_.inference_frame;

    const double sector_center_x = params_.detection_area_center_x;
    const double sector_center_y = params_.detection_area_center_y;
    const double sector_radius = params_.detection_area_radius;
    const double sector_bearing_deg = params_.detection_area_bearing_deg;
    const double sector_fov_deg = params_.detection_area_fov_deg;
    const int num_segments = std::max(3, params_.detection_area_num_segments);

    // Build sector polygon: start at center, then arc points, then back to center
    std::vector<geometry_msgs::msg::Point32> sector_poly;
    {
      geometry_msgs::msg::Point32 pt;
      pt.x = static_cast<float>(sector_center_x);
      pt.y = static_cast<float>(sector_center_y);
      pt.z = 0.0f;
      sector_poly.push_back(pt);

      const double start_angle_deg = sector_bearing_deg - 0.5 * sector_fov_deg;
      const double end_angle_deg = sector_bearing_deg + 0.5 * sector_fov_deg;
      for (int i = 0; i <= num_segments; ++i) {
        const double angle_deg =
            start_angle_deg + (end_angle_deg - start_angle_deg) * static_cast<double>(i) / num_segments;
        const double angle_rad = angle_deg * M_PI / 180.0;
        geometry_msgs::msg::Point32 arc_point;
        arc_point.x = static_cast<float>(sector_center_x + sector_radius * std::cos(angle_rad));
        arc_point.y = static_cast<float>(sector_center_y + sector_radius * std::sin(angle_rad));
        arc_point.z = 0.0f;
        sector_poly.push_back(arc_point);
      }
    }

    // Sutherland–Hodgman clipping of polygon to model bounds rectangle
    auto clip_poly_to_halfplane = [](const std::vector<geometry_msgs::msg::Point32>& poly, auto inside,
                                     auto intersect) {
      std::vector<geometry_msgs::msg::Point32> output;
      if (poly.empty()) return output;
      for (size_t i = 0; i < poly.size(); ++i) {
        const auto& current = poly[i];
        const auto& prev = poly[(i + poly.size() - 1) % poly.size()];
        const bool curr_in = inside(current);
        const bool prev_in = inside(prev);
        if (curr_in) {
          if (!prev_in) {
            output.push_back(intersect(prev, current));
          }
          output.push_back(current);
        } else if (prev_in) {
          output.push_back(intersect(prev, current));
        }
      }
      return output;
    };

    const double rx_min = static_cast<double>(model_config_.x_min);
    const double rx_max = static_cast<double>(model_config_.x_max);
    const double ry_min = static_cast<double>(model_config_.y_min);
    const double ry_max = static_cast<double>(model_config_.y_max);

    // Intersections with vertical and horizontal lines
    auto intersect_x = [](const geometry_msgs::msg::Point32& a, const geometry_msgs::msg::Point32& b, double x_val) {
      geometry_msgs::msg::Point32 out;
      const double dx = static_cast<double>(b.x) - static_cast<double>(a.x);
      const double t = (std::abs(dx) < 1e-9) ? 0.0 : (x_val - static_cast<double>(a.x)) / dx;
      out.x = static_cast<float>(x_val);
      out.y = static_cast<float>(static_cast<double>(a.y) + t * (static_cast<double>(b.y) - static_cast<double>(a.y)));
      out.z = 0.0f;
      return out;
    };
    auto intersect_y = [](const geometry_msgs::msg::Point32& a, const geometry_msgs::msg::Point32& b, double y_val) {
      geometry_msgs::msg::Point32 out;
      const double dy = static_cast<double>(b.y) - static_cast<double>(a.y);
      const double t = (std::abs(dy) < 1e-9) ? 0.0 : (y_val - static_cast<double>(a.y)) / dy;
      out.y = static_cast<float>(y_val);
      out.x = static_cast<float>(static_cast<double>(a.x) + t * (static_cast<double>(b.x) - static_cast<double>(a.x)));
      out.z = 0.0f;
      return out;
    };

    // Clip in order: left, right, bottom, top
    std::vector<geometry_msgs::msg::Point32> clipped = sector_poly;
    clipped = clip_poly_to_halfplane(
        clipped, [&](const auto& p) { return static_cast<double>(p.x) >= rx_min; },
        [&](const auto& a, const auto& b) { return intersect_x(a, b, rx_min); });
    clipped = clip_poly_to_halfplane(
        clipped, [&](const auto& p) { return static_cast<double>(p.x) <= rx_max; },
        [&](const auto& a, const auto& b) { return intersect_x(a, b, rx_max); });
    clipped = clip_poly_to_halfplane(
        clipped, [&](const auto& p) { return static_cast<double>(p.y) >= ry_min; },
        [&](const auto& a, const auto& b) { return intersect_y(a, b, ry_min); });
    clipped = clip_poly_to_halfplane(
        clipped, [&](const auto& p) { return static_cast<double>(p.y) <= ry_max; },
        [&](const auto& a, const auto& b) { return intersect_y(a, b, ry_max); });

    // Publish the sector polygon clipped only to model bounds.
    // Note: Do not warp or clip against the no-detection zone; visualization
    // polygons should be independent to avoid confusing overlays.
    if (!clipped.empty()) {
      polygon_message.polygon.points = clipped;
      detection_area_pub_->publish(polygon_message);
    }
  }

  // pointcloudToBoxes
  // index: 2 & 3, inside model call
  std::vector<BoundingBox> center_boxes = (*detection_model_)(point_cloud, timestamps);
  timestamps.push_back(std::chrono::high_resolution_clock::now());  // index: 4, after output tensor creation

  // non-maxmimum suppresion
  if (model_type_ != ModelType::TPOD) {
    non_max_suppression_->nms(center_boxes);
  }
  timestamps.push_back(std::chrono::high_resolution_clock::now());  // index: 5, after nms

  // filter detections intersecting no-detection zone (inference_frame)
  if (params_.no_detection_zone_enabled) {
    BoundingBox no_detection_rectangle;
    const float rect_x_min = static_cast<float>(params_.no_detection_zone_x_min);
    const float rect_x_max = static_cast<float>(params_.no_detection_zone_x_max);
    const float rect_y_min = static_cast<float>(params_.no_detection_zone_y_min);
    const float rect_y_max = static_cast<float>(params_.no_detection_zone_y_max);
    no_detection_rectangle.center = {0.5f * (rect_x_min + rect_x_max), 0.5f * (rect_y_min + rect_y_max)};
    no_detection_rectangle.z = 0.0f;
    no_detection_rectangle.length = (rect_x_max - rect_x_min);
    no_detection_rectangle.width = (rect_y_max - rect_y_min);
    no_detection_rectangle.height = 0.0f;  // not used for 2D intersection
    no_detection_rectangle.yaw = 0.0f;     // axis-aligned rectangle in inference_frame
    no_detection_rectangle.existence_probability = 1.0f;

    auto num_boxes_before = center_boxes.size();
    center_boxes.erase(std::remove_if(center_boxes.begin(), center_boxes.end(),
                                      [&no_detection_rectangle](const BoundingBox& bbox) {
                                        return bbox.intersection_area(no_detection_rectangle) > 0.0f;
                                      }),
                       center_boxes.end());
    auto removed = num_boxes_before - center_boxes.size();
    if (removed > 0) {
      RCLCPP_INFO(this->get_logger(), "Filtered %zu detections inside no-detection zone", removed);
    }
  }

  // Filter detections outside the detection area sector (inference_frame)
  if (params_.detection_area_enabled && params_.detection_area_filter_detections) {
    const double sector_center_x = params_.detection_area_center_x;
    const double sector_center_y = params_.detection_area_center_y;
    const double sector_radius = params_.detection_area_radius;
    const double sector_bearing_rad = params_.detection_area_bearing_deg * M_PI / 180.0;
    const double sector_fov_rad = params_.detection_area_fov_deg * M_PI / 180.0;
    const bool require_complete_box_inside = (params_.detection_area_filter_mode == "complete");

    auto is_point_inside_sector = [&](double x, double y) -> bool {
      const double delta_x = x - sector_center_x;
      const double delta_y = y - sector_center_y;
      const double squared_distance_to_center = delta_x * delta_x + delta_y * delta_y;
      if (squared_distance_to_center > sector_radius * sector_radius) return false;
      double angle_to_center = std::atan2(delta_y, delta_x);
      double angle_offset = angle_to_center - sector_bearing_rad;
      // wrap to [-pi, pi]
      while (angle_offset > M_PI) angle_offset -= 2.0 * M_PI;
      while (angle_offset < -M_PI) angle_offset += 2.0 * M_PI;
      return std::abs(angle_offset) <= (sector_fov_rad * 0.5 + 1e-9);
    };

    auto is_bounding_box_inside_sector = [&](const BoundingBox& bbox) -> bool {
      if (!require_complete_box_inside) {
        return is_point_inside_sector(bbox.center[0], bbox.center[1]);
      }
      // Check all 4 rectangle vertices in 2D
      auto vertices = bbox.rectangle_vertices();
      for (const auto& vertex : vertices) {
        if (!is_point_inside_sector(vertex.x, vertex.y)) return false;
      }
      return true;
    };

    auto num_boxes_before = center_boxes.size();
    center_boxes.erase(std::remove_if(center_boxes.begin(), center_boxes.end(),
                                      [&](const BoundingBox& bbox) { return !is_bounding_box_inside_sector(bbox); }),
                       center_boxes.end());
    auto removed = num_boxes_before - center_boxes.size();
    if (removed > 0) {
      RCLCPP_INFO(this->get_logger(), "Filtered %zu detections outside detection area (%s)", removed,
                  params_.detection_area_filter_mode.c_str());
    }
  }

  // publish class-specific and unclassified point clouds
  publishClassPointClouds(center_boxes, header, point_cloud);

  // boxesToObjectList
  pm::msg::ObjectList::UniquePtr object_list = std::make_unique<pm::msg::ObjectList>();
  object_list->header = header;
  boxesToObjectList(center_boxes, *object_list);
  timestamps.push_back(std::chrono::high_resolution_clock::now());  // index: 6, after nms

  // get size for logging, as publishing invalidates the message here
  std::size_t size = object_list->objects.size();

  // publish output message
  publisher_->publish(std::move(object_list));
  timestamps.push_back(std::chrono::high_resolution_clock::now());  // index: 7, after nms

  // log processing
  std::chrono::duration<double> inference_time = timestamps[3] - timestamps[2];
  std::chrono::duration<double> total_time = timestamps.back() - timestamps.front();
  RCLCPP_INFO(this->get_logger(), "%ld objects detected in %.3fs (inference %.3fs)", size, total_time.count(),
              inference_time.count());
}

// Transition callback for state configuring
// Lifecycle callbacks removed in regular node
}  // namespace point_cloud_object_detection

RCLCPP_COMPONENTS_REGISTER_NODE(point_cloud_object_detection::PointCloudObjectDetection)
