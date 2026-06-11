// Copyright Institute for Automotive Engineering (ika), RWTH Aachen University
// SPDX-License-Identifier: Apache-2.0

#include "point_cloud_object_detection/PointCloudObjectDetection.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <regex>
#include <sstream>
#include <utility>

#include <boost/algorithm/string/case_conv.hpp>

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <rclcpp/logging.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <sensor_msgs/msg/point_field.hpp>
#include <stdexcept>

#include "pcod_common/model_manifest.hpp"
#include "point_cloud_object_detection/Utils.hpp"

namespace point_cloud_object_detection {

namespace {

[[noreturn]] void throwParameterError(const rclcpp::Logger& logger, const std::string& param_name, const std::string& details) {
  RCLCPP_FATAL(logger, "Invalid parameter '%s': %s", param_name.c_str(), details.c_str());
  throw std::runtime_error("Invalid parameter '" + param_name + "': " + details);
}

bool isProbability(double value) { return std::isfinite(value) && value >= 0.0 && value <= 1.0; }

bool isFinite(double value) { return std::isfinite(value); }

bool isAllowedPointFeatureField(const std::string& source) {
  return std::any_of(kAllowedPointFeatureFields.begin(), kAllowedPointFeatureFields.end(),
                     [&](const char* allowed) { return source == allowed; });
}

bool isAllowedPreprocessingBackend(const std::string& backend) {
  return std::any_of(kAllowedPreprocessingBackends.begin(), kAllowedPreprocessingBackends.end(),
                     [&](const char* allowed) { return backend == allowed; });
}

std::string allowedPointFeatureFieldsString() {
  std::ostringstream oss;
  for (std::size_t i = 0; i < kAllowedPointFeatureFields.size(); ++i) {
    if (i > 0) {
      oss << ", ";
    }
    oss << "'" << kAllowedPointFeatureFields[i] << "'";
  }
  return oss.str();
}

std::string allowedPreprocessingBackendsString() {
  std::ostringstream oss;
  for (std::size_t i = 0; i < kAllowedPreprocessingBackends.size(); ++i) {
    if (i > 0) {
      oss << ", ";
    }
    oss << "'" << kAllowedPreprocessingBackends[i] << "'";
  }
  return oss.str();
}

bool isSupportedManifestPrecision(const std::string& precision) {
  return std::any_of(kSupportedManifestPrecisions.begin(), kSupportedManifestPrecisions.end(),
                     [&](const char* supported) { return precision == supported; });
}

// Keep this list aligned with any runtime parameter that is cached inside PBODModel.
// If a parameter influences model-side cached state, either add it here for reinit
// or update the live model instance explicitly in the parameter callback.
bool requiresModelReinitializationForParameter(const std::string& name) {
  return name == "prediction.triton_client_timeout_s" || name == "prediction.use_shm" || name == "prediction.cuda_input_shm" ||
         name == "preprocessing.backend" || name == "preprocessing.point_feature.value_threshold" ||
         name == "preprocessing.detection_area.z_min" || name == "preprocessing.detection_area.z_max" ||
         name == "postprocessing.nms.score_threshold" || name == "preprocessing.no_detection_zone.remove_points" ||
         name == "preprocessing.no_detection_zone.x_min" || name == "preprocessing.no_detection_zone.x_max" ||
         name == "preprocessing.no_detection_zone.y_min" || name == "preprocessing.no_detection_zone.y_max" ||
         name == "preprocessing.detection_area.enabled" || name == "preprocessing.detection_area.center_x" ||
         name == "preprocessing.detection_area.center_y" || name == "preprocessing.detection_area.radius" ||
         name == "preprocessing.detection_area.bearing_deg" || name == "preprocessing.detection_area.fov_deg" ||
         name == "preprocessing.detection_area.num_segments" || name == "prediction.model_repository" ||
         name == "prediction.model_version";
}

std::string resolveModelRepositoryPath(const std::string& path) {
  if (path.empty()) {
    return path;
  }
  const std::filesystem::path repository_path(path);
  if (repository_path.is_absolute()) {
    return path;
  }

  const std::filesystem::path models_dir = "/models";
  if (std::filesystem::exists(models_dir / repository_path)) {
    return (models_dir / repository_path).string();
  }

  const auto share_dir = ament_index_cpp::get_package_share_directory("point_cloud_object_detection");
  return (std::filesystem::path(share_dir) / repository_path).string();
}

std::string manifestPathFromRepository(const std::string& repository_path) {
  if (repository_path.empty()) {
    return repository_path;
  }
  return (std::filesystem::path(repository_path) / "model_manifest.yml").string();
}

std::string parseTritonModelNameFromConfig(const std::string& config_path) {
  std::ifstream config_stream(config_path);
  if (!config_stream) {
    throw std::runtime_error("failed to open config.pbtxt");
  }

  static const std::regex kNamePattern(R"pbtxt(^\s*name\s*:\s*"([^"]+)"\s*$)pbtxt");
  std::string line;
  while (std::getline(config_stream, line)) {
    std::smatch match;
    if (std::regex_match(line, match, kNamePattern)) {
      return match[1].str();
    }
  }

  throw std::runtime_error("config.pbtxt does not define a top-level name field");
}

std::string resolvePrimaryFeatureField(const ModelConfig& model_config, const Params& params) {
  // Kept model-aware on purpose: future model families may map feature channels differently.
  (void)model_config;
  return params.point_feature_field;
}

void sanitizeVarianceVector(std::vector<double>& variance, double sentinel) {
  // Accept slightly-off "-1" encodings from YAML/ROS transport and normalize to the exact sentinel.
  for (double& value : variance) {
    if (value < 0.0 && std::fabs(value + 1.0) <= 1e-9) {
      value = sentinel;
    }
  }
}

const char* pointFieldDatatypeToString(uint8_t datatype) {
  using PF = sensor_msgs::msg::PointField;
  switch (datatype) {
    case PF::INT8:
      return "INT8";
    case PF::UINT8:
      return "UINT8";
    case PF::INT16:
      return "INT16";
    case PF::UINT16:
      return "UINT16";
    case PF::INT32:
      return "INT32";
    case PF::UINT32:
      return "UINT32";
    case PF::FLOAT32:
      return "FLOAT32";
    case PF::FLOAT64:
      return "FLOAT64";
    default:
      return "UNKNOWN";
  }
}

bool isHostLittleEndian() {
  const std::uint16_t value = 1;
  return *reinterpret_cast<const std::uint8_t*>(&value) == 1;
}

constexpr double kAuxiliaryGridMapUnityGainEpsilon = 1e-6;
constexpr double kAuxiliaryGridMapResolutionTolerance = 1e-6;

bool shouldApplyAuxiliaryGridMapGain(double gain) { return std::fabs(gain - 1.0) > kAuxiliaryGridMapUnityGainEpsilon; }

float applyAuxiliaryGridMapGainToValue(float value, double gain) {
  if (shouldApplyAuxiliaryGridMapGain(gain)) {
    value *= static_cast<float>(gain);
  }
  return clamp(value, 0.0f, 1.0f);
}

bool haveCompatibleAuxiliaryGridMaps(const AuxiliaryGridMap& first, const AuxiliaryGridMap& second) {
  return first.grid_x == second.grid_x && first.grid_y == second.grid_y && first.values.size() == second.values.size();
}

AuxiliaryGridMap makeAuxiliaryGridMapLike(const AuxiliaryGridMap& reference, const char* layer) {
  return AuxiliaryGridMap{layer,
                          std::vector<float>(reference.values.size()),
                          reference.grid_x,
                          reference.grid_y,
                          reference.x_min,
                          reference.x_max,
                          reference.y_min,
                          reference.y_max};
}

std::optional<nav_msgs::msg::OccupancyGrid> initializeAuxiliaryOccupancyGridMessage(const AuxiliaryGridMap& grid_map_output,
                                                                                    const std_msgs::msg::Header& header,
                                                                                    const std::string& frame_id,
                                                                                    const geometry_msgs::msg::Pose& origin) {
  if (grid_map_output.grid_x <= 0 || grid_map_output.grid_y <= 0) {
    return std::nullopt;
  }
  if (grid_map_output.values.size() != static_cast<std::size_t>(grid_map_output.grid_x * grid_map_output.grid_y)) {
    return std::nullopt;
  }

  const double x_length = static_cast<double>(grid_map_output.x_max) - static_cast<double>(grid_map_output.x_min);
  const double y_length = static_cast<double>(grid_map_output.y_max) - static_cast<double>(grid_map_output.y_min);
  if (!(x_length > 0.0) || !(y_length > 0.0)) {
    return std::nullopt;
  }

  const double resolution_x = x_length / static_cast<double>(grid_map_output.grid_x);
  const double resolution_y = y_length / static_cast<double>(grid_map_output.grid_y);
  if (std::fabs(resolution_x - resolution_y) > kAuxiliaryGridMapResolutionTolerance) {
    return std::nullopt;
  }

  nav_msgs::msg::OccupancyGrid message;
  message.header = header;
  message.header.frame_id = frame_id;
  message.info.map_load_time = header.stamp;
  message.info.resolution = static_cast<float>(resolution_x);
  message.info.width = static_cast<std::uint32_t>(grid_map_output.grid_x);
  message.info.height = static_cast<std::uint32_t>(grid_map_output.grid_y);
  message.info.origin = origin;
  message.data.assign(static_cast<std::size_t>(grid_map_output.grid_x * grid_map_output.grid_y), 0);
  return message;
}

int8_t toOccupancyGridCellValue(float value, double gain = 1.0) {
  return static_cast<int8_t>(std::lround(applyAuxiliaryGridMapGainToValue(value, gain) * 100.0f));
}

std::optional<nav_msgs::msg::OccupancyGrid> buildAuxiliaryOccupancyGridMessage(const AuxiliaryGridMap& grid_map_output,
                                                                               const std_msgs::msg::Header& header,
                                                                               const std::string& frame_id,
                                                                               const geometry_msgs::msg::Pose& origin) {
  auto message = initializeAuxiliaryOccupancyGridMessage(grid_map_output, header, frame_id, origin);
  if (!message.has_value()) {
    return std::nullopt;
  }

  for (int ix = 0; ix < grid_map_output.grid_x; ++ix) {
    for (int iy = 0; iy < grid_map_output.grid_y; ++iy) {
      const std::size_t source_index = static_cast<std::size_t>(ix * grid_map_output.grid_y + iy);
      const std::size_t target_index = static_cast<std::size_t>(iy * grid_map_output.grid_x + ix);
      message->data[target_index] = toOccupancyGridCellValue(grid_map_output.values[source_index]);
    }
  }

  return message;
}

geometry_msgs::msg::Pose makeAuxiliaryGridMapOriginPose(const AuxiliaryGridMap& grid_map_output) {
  geometry_msgs::msg::Pose origin;
  origin.position.x = static_cast<double>(grid_map_output.x_min);
  origin.position.y = static_cast<double>(grid_map_output.y_min);
  origin.position.z = 0.0;
  origin.orientation.w = 1.0;
  return origin;
}

std::string resolveAuxiliaryGridMapSourceFrame(const Params& params, const std_msgs::msg::Header& header) {
  return params.inference_frame.empty() ? header.frame_id : params.inference_frame;
}

std::string resolveAuxiliaryGridMapTargetFrame(const Params& params, const std::string& source_frame) {
  return params.grid_map_frame.empty() ? source_frame : params.grid_map_frame;
}

std::optional<geometry_msgs::msg::Pose> transformAuxiliaryGridMapOriginPose(const geometry_msgs::msg::Pose& source_origin,
                                                                            const std_msgs::msg::Header& header,
                                                                            const std::string& source_frame,
                                                                            const std::string& target_frame,
                                                                            tf2_ros::Buffer& tf_buffer,
                                                                            const rclcpp::Logger& logger) {
  if (target_frame.empty() || target_frame == source_frame) {
    return source_origin;
  }

  geometry_msgs::msg::TransformStamped transform;
  try {
    transform = tf_buffer.lookupTransform(target_frame, source_frame, header.stamp);
  } catch (tf2::TransformException& e) {
    RCLCPP_ERROR(logger, "Cannot transform grid map: Transformation from source frame (%s) to output frame (%s) not found: %s",
                 source_frame.c_str(), target_frame.c_str(), e.what());
    return std::nullopt;
  }

  const auto& translation = transform.transform.translation;
  const auto& rotation = transform.transform.rotation;
  tf2::Quaternion target_from_source_rotation(rotation.x, rotation.y, rotation.z, rotation.w);
  target_from_source_rotation.normalize();
  const tf2::Transform target_from_source(target_from_source_rotation, tf2::Vector3(translation.x, translation.y, translation.z));

  const auto& source_position = source_origin.position;
  const auto& source_orientation = source_origin.orientation;
  tf2::Quaternion source_origin_rotation(source_orientation.x, source_orientation.y, source_orientation.z, source_orientation.w);
  source_origin_rotation.normalize();
  const tf2::Transform source_from_grid(source_origin_rotation,
                                        tf2::Vector3(source_position.x, source_position.y, source_position.z));

  const tf2::Transform target_from_grid = target_from_source * source_from_grid;

  geometry_msgs::msg::Pose target_origin;
  const tf2::Vector3 target_position = target_from_grid.getOrigin();
  target_origin.position.x = target_position.x();
  target_origin.position.y = target_position.y();
  target_origin.position.z = target_position.z();
  const tf2::Quaternion target_orientation = target_from_grid.getRotation();
  target_origin.orientation.x = target_orientation.x();
  target_origin.orientation.y = target_orientation.y();
  target_origin.orientation.z = target_orientation.z();
  target_origin.orientation.w = target_orientation.w();
  return target_origin;
}

std::optional<AuxiliaryGridMap> buildCombinedAuxiliaryGridMap(const AuxiliaryGridMap& density_grid_map,
                                                              const AuxiliaryGridMap& occupancy_grid_map) {
  if (!haveCompatibleAuxiliaryGridMaps(density_grid_map, occupancy_grid_map)) {
    return std::nullopt;
  }

  AuxiliaryGridMap combined_grid_map = makeAuxiliaryGridMapLike(density_grid_map, "combined");
  for (std::size_t i = 0; i < combined_grid_map.values.size(); ++i) {
    const float density = clamp(density_grid_map.values[i], 0.0f, 1.0f);
    const float occupancy = clamp(occupancy_grid_map.values[i], 0.0f, 1.0f);
    combined_grid_map.values[i] = occupancy + 0.5f * density * (1.0f - occupancy);
  }
  return combined_grid_map;
}

void applyAuxiliaryGridMapGain(AuxiliaryGridMap& grid_map_output, double gain) {
  if (!shouldApplyAuxiliaryGridMapGain(gain) || grid_map_output.values.empty()) {
    return;
  }

  const float gain_f = static_cast<float>(gain);
  for (float& value : grid_map_output.values) {
    value = clamp(value * gain_f, 0.0f, 1.0f);
  }
}

std::optional<AuxiliaryGridMap> buildStaticAuxiliaryGridMap(const AuxiliaryGridMap& density_grid_map,
                                                            const AuxiliaryGridMap& occupancy_grid_map) {
  if (!haveCompatibleAuxiliaryGridMaps(density_grid_map, occupancy_grid_map)) {
    return std::nullopt;
  }

  AuxiliaryGridMap static_grid_map = makeAuxiliaryGridMapLike(density_grid_map, "static");
  for (std::size_t i = 0; i < static_grid_map.values.size(); ++i) {
    const float density = clamp(density_grid_map.values[i], 0.0f, 1.0f);
    const float occupancy = clamp(occupancy_grid_map.values[i], 0.0f, 1.0f);
    static_grid_map.values[i] = density * (1.0f - occupancy);
  }
  return static_grid_map;
}

constexpr double kGridMapMaskMinDetectionAreaRadius = 0.0;
constexpr double kGridMapMaskMinDetectionAreaFovDeg = 0.0;
constexpr double kGridMapMaskMaxDetectionAreaFovDeg = 360.0;

bool isPointInsideDetectionAreaSector(double x, double y, const Params& params) {
  const double sector_center_x = params.detection_area_center_x;
  const double sector_center_y = params.detection_area_center_y;
  const double sector_radius = params.detection_area_radius;
  const double sector_bearing_rad = params.detection_area_bearing_deg * M_PI / 180.0;
  const double sector_fov_rad = params.detection_area_fov_deg * M_PI / 180.0;
  constexpr double kFullCircleToleranceDeg = 1e-3;
  const bool is_full_circle =
      std::abs(params.detection_area_fov_deg - kGridMapMaskMaxDetectionAreaFovDeg) <= kFullCircleToleranceDeg;

  const double delta_x = x - sector_center_x;
  const double delta_y = y - sector_center_y;
  const double squared_distance_to_center = delta_x * delta_x + delta_y * delta_y;
  if (squared_distance_to_center > sector_radius * sector_radius) {
    return false;
  }
  if (is_full_circle) {
    return true;
  }

  const double angle_to_center = std::atan2(delta_y, delta_x);
  double angle_offset = angle_to_center - sector_bearing_rad;
  while (angle_offset > M_PI) angle_offset -= 2.0 * M_PI;
  while (angle_offset < -M_PI) angle_offset += 2.0 * M_PI;
  return std::abs(angle_offset) <= (sector_fov_rad * 0.5 + 1e-9);
}

void applyAuxiliaryGridMapMasks(AuxiliaryGridMap& grid_map_output, const Params& params) {
  if (grid_map_output.grid_x <= 0 || grid_map_output.grid_y <= 0 || grid_map_output.values.empty()) {
    return;
  }

  const bool zero_in_no_detection_zone = params.zero_grid_map_cells_in_no_detection_zone && params.no_detection_zone_enabled;
  const bool zero_outside_detection_area = params.zero_grid_map_cells_outside_detection_area && params.detection_area_enabled &&
                                           params.detection_area_radius > kGridMapMaskMinDetectionAreaRadius &&
                                           params.detection_area_fov_deg > kGridMapMaskMinDetectionAreaFovDeg;
  if (!zero_in_no_detection_zone && !zero_outside_detection_area) {
    return;
  }

  const double resolution_x =
      (static_cast<double>(grid_map_output.x_max) - static_cast<double>(grid_map_output.x_min)) / grid_map_output.grid_x;
  const double resolution_y =
      (static_cast<double>(grid_map_output.y_max) - static_cast<double>(grid_map_output.y_min)) / grid_map_output.grid_y;

  for (int ix = 0; ix < grid_map_output.grid_x; ++ix) {
    for (int iy = 0; iy < grid_map_output.grid_y; ++iy) {
      const double x = static_cast<double>(grid_map_output.x_min) + (static_cast<double>(ix) + 0.5) * resolution_x;
      const double y = static_cast<double>(grid_map_output.y_min) + (static_cast<double>(iy) + 0.5) * resolution_y;

      const bool in_no_detection_zone = zero_in_no_detection_zone && x >= params.no_detection_zone_x_min &&
                                        x <= params.no_detection_zone_x_max && y >= params.no_detection_zone_y_min &&
                                        y <= params.no_detection_zone_y_max;
      const bool outside_detection_area = zero_outside_detection_area && !isPointInsideDetectionAreaSector(x, y, params);

      if (in_no_detection_zone || outside_detection_area) {
        const std::size_t flat_index = static_cast<std::size_t>(ix * grid_map_output.grid_y + iy);
        grid_map_output.values[flat_index] = 0.0f;
      }
    }
  }
}

std::uint16_t byteSwap16(std::uint16_t value) { return static_cast<std::uint16_t>((value >> 8) | (value << 8)); }

std::uint32_t byteSwap32(std::uint32_t value) {
  return ((value & 0x000000FFu) << 24) | ((value & 0x0000FF00u) << 8) | ((value & 0x00FF0000u) >> 8) |
         ((value & 0xFF000000u) >> 24);
}

std::uint64_t byteSwap64(std::uint64_t value) {
  return ((value & 0x00000000000000FFull) << 56) | ((value & 0x000000000000FF00ull) << 40) |
         ((value & 0x0000000000FF0000ull) << 24) | ((value & 0x00000000FF000000ull) << 8) |
         ((value & 0x000000FF00000000ull) >> 8) | ((value & 0x0000FF0000000000ull) >> 24) |
         ((value & 0x00FF000000000000ull) >> 40) | ((value & 0xFF00000000000000ull) >> 56);
}

float readFloat32At(const std::uint8_t* src, bool needs_swap) {
  // Use memcpy for aliasing/alignment-safe bit reinterpretation from packed PointCloud2 buffers.
  std::uint32_t bits = 0;
  std::memcpy(&bits, src, sizeof(bits));
  if (needs_swap) bits = byteSwap32(bits);
  float out = 0.0f;
  std::memcpy(&out, &bits, sizeof(out));
  return out;
}

float readFloat64AsFloatAt(const std::uint8_t* src, bool needs_swap) {
  std::uint64_t bits = 0;
  std::memcpy(&bits, src, sizeof(bits));
  if (needs_swap) bits = byteSwap64(bits);
  double out = 0.0;
  std::memcpy(&out, &bits, sizeof(out));
  return static_cast<float>(out);
}

float readPointFieldAsFloat(const std::uint8_t* src, std::uint8_t datatype, bool needs_swap) {
  using PF = sensor_msgs::msg::PointField;
  switch (datatype) {
    case PF::FLOAT32:
      return readFloat32At(src, needs_swap);
    case PF::FLOAT64:
      return readFloat64AsFloatAt(src, needs_swap);
    case PF::UINT16: {
      std::uint16_t value = 0;
      std::memcpy(&value, src, sizeof(value));
      if (needs_swap) value = byteSwap16(value);
      return static_cast<float>(value);
    }
    case PF::UINT8:
      return static_cast<float>(*src);
    case PF::INT16: {
      std::uint16_t raw = 0;
      std::memcpy(&raw, src, sizeof(raw));
      if (needs_swap) raw = byteSwap16(raw);
      return static_cast<float>(static_cast<std::int16_t>(raw));
    }
    case PF::INT8:
      return static_cast<float>(static_cast<std::int8_t>(*src));
    case PF::UINT32: {
      std::uint32_t value = 0;
      std::memcpy(&value, src, sizeof(value));
      if (needs_swap) value = byteSwap32(value);
      return static_cast<float>(value);
    }
    case PF::INT32: {
      std::uint32_t raw = 0;
      std::memcpy(&raw, src, sizeof(raw));
      if (needs_swap) raw = byteSwap32(raw);
      return static_cast<float>(static_cast<std::int32_t>(raw));
    }
    default:
      throw std::runtime_error("Unsupported PointCloud2 datatype");
  }
}

struct PreparedPointCloudInput {
  sensor_msgs::msg::PointCloud2::ConstSharedPtr msg;
  sensor_msgs::msg::PointCloud2::SharedPtr transformed_owner;
  uint32_t x_offset = 0;
  uint32_t y_offset = 0;
  uint32_t z_offset = 0;
  uint32_t feature_offset = 0;
  uint8_t feature_datatype = sensor_msgs::msg::PointField::FLOAT32;
  bool needs_swap = false;
  std::size_t total_points = 0;
};

PreparedPointCloudInput preparePointCloudInput(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& msg,
                                               const ModelConfig& model_config,
                                               const Params& params,
                                               tf2_ros::Buffer& tf_buffer,
                                               const rclcpp::Logger& logger) {
  PreparedPointCloudInput prepared;
  prepared.msg = msg;

  if (!params.inference_frame.empty() && msg->header.frame_id != params.inference_frame) {
    prepared.transformed_owner = std::make_shared<sensor_msgs::msg::PointCloud2>();
    try {
      tf_buffer.transform(*msg, *prepared.transformed_owner, params.inference_frame);
      prepared.msg = prepared.transformed_owner;
    } catch (tf2::TransformException& e) {
      RCLCPP_ERROR(logger,
                   "Cannot tranform Pointcloud: Transformation from its frame (%s) to inference_frame "
                   "(%s) not found: %s",
                   msg->header.frame_id.c_str(), params.inference_frame.c_str(), e.what());
      throw;
    }
  }

  auto findField = [&](const std::string& name) -> const sensor_msgs::msg::PointField* {
    for (const auto& field : prepared.msg->fields) {
      if (field.name == name) {
        return &field;
      }
    }
    return nullptr;
  };

  const std::string primary_feature_field = resolvePrimaryFeatureField(model_config, params);
  const auto* x_field = findField("x");
  const auto* y_field = findField("y");
  const auto* z_field = findField("z");
  const auto* primary_feature = findField(primary_feature_field);

  if (!x_field || !y_field || !z_field) {
    RCLCPP_FATAL(logger, "Point fields 'x', 'y', 'z' are required in PointCloud2 input");
    throw std::runtime_error("Missing required PointCloud2 coordinate fields");
  }
  if (!primary_feature) {
    RCLCPP_FATAL(logger, "Point field '%s' is required for primary feature extraction", primary_feature_field.c_str());
    throw std::runtime_error("Missing required PointCloud2 field: " + primary_feature_field);
  }

  using PF = sensor_msgs::msg::PointField;
  if (x_field->datatype != PF::FLOAT32 || y_field->datatype != PF::FLOAT32 || z_field->datatype != PF::FLOAT32) {
    RCLCPP_FATAL(logger, "Point fields 'x', 'y', 'z' must be FLOAT32");
    throw std::runtime_error("Unsupported PointCloud2 datatype for x/y/z fields");
  }

  switch (primary_feature->datatype) {
    case PF::FLOAT32:
      break;
    case PF::FLOAT64:
    case PF::UINT16:
    case PF::UINT8:
    case PF::INT16:
    case PF::INT8:
    case PF::UINT32:
    case PF::INT32:
      RCLCPP_WARN_ONCE(logger, "Converting PointCloud2 field '%s' from %s to FLOAT32", primary_feature_field.c_str(),
                       pointFieldDatatypeToString(primary_feature->datatype));
      break;
    default:
      RCLCPP_FATAL(logger,
                   "Point field '%s' has unsupported datatype %u (%s). Supported: INT8, UINT8, INT16, UINT16, "
                   "INT32, UINT32, FLOAT32, FLOAT64",
                   primary_feature_field.c_str(), primary_feature->datatype,
                   pointFieldDatatypeToString(primary_feature->datatype));
      throw std::runtime_error("Unsupported PointCloud2 datatype for field: " + primary_feature_field);
  }

  prepared.x_offset = x_field->offset;
  prepared.y_offset = y_field->offset;
  prepared.z_offset = z_field->offset;
  prepared.feature_offset = primary_feature->offset;
  prepared.feature_datatype = primary_feature->datatype;
  prepared.needs_swap = prepared.msg->is_bigendian == isHostLittleEndian();
  prepared.total_points = static_cast<std::size_t>(prepared.msg->width) * prepared.msg->height;
  return prepared;
}

void decodePreparedPointCloudToPcl(const PreparedPointCloudInput& prepared, PointCloud& point_cloud) {
  const auto& msg_ref = *prepared.msg;
  point_cloud.clear();
  pcl_conversions::toPCL(msg_ref.header, point_cloud.header);
  point_cloud.width = msg_ref.width;
  point_cloud.height = msg_ref.height;
  point_cloud.is_dense = msg_ref.is_dense;
  point_cloud.points.resize(prepared.total_points);

  std::size_t out_idx = 0;
  const std::size_t row_step = static_cast<std::size_t>(msg_ref.row_step);
  const std::size_t point_step = static_cast<std::size_t>(msg_ref.point_step);
  for (std::size_t row = 0; row < msg_ref.height; ++row) {
    const std::uint8_t* row_ptr = msg_ref.data.data() + row * row_step;
    for (std::size_t col = 0; col < msg_ref.width; ++col) {
      const std::uint8_t* point_ptr = row_ptr + col * point_step;
      auto& dst = point_cloud.points[out_idx++];
      dst.x = readFloat32At(point_ptr + prepared.x_offset, prepared.needs_swap);
      dst.y = readFloat32At(point_ptr + prepared.y_offset, prepared.needs_swap);
      dst.z = readFloat32At(point_ptr + prepared.z_offset, prepared.needs_swap);
      dst.intensity = readPointFieldAsFloat(point_ptr + prepared.feature_offset, prepared.feature_datatype, prepared.needs_swap);
    }
  }
}

}  // namespace

// constants
const std::string PointCloudObjectDetection::kInputTopic = "~/point_cloud";
const std::string PointCloudObjectDetection::kOutputTopic = "~/object_list";
const std::string PointCloudObjectDetection::kNoDetectionZoneTopic = "~/no_detection_zone";
const std::string PointCloudObjectDetection::kNoDetectionZonePointsTopic = "~/no_detection_zone_points";
const std::string PointCloudObjectDetection::kDetectionAreaTopic = "~/detection_area";
const std::string PointCloudObjectDetection::kModelBoundsTopic = "~/model_bounds";
const std::string PointCloudObjectDetection::kDensityGridMapTopic = "~/density_grid_map";
const std::string PointCloudObjectDetection::kOccupancyGridMapTopic = "~/occupancy_grid_map";
const std::string PointCloudObjectDetection::kCombinedGridMapTopic = "~/combined_grid_map";
const std::string PointCloudObjectDetection::kStaticGridMapTopic = "~/static_grid_map";
const std::map<uint8_t, std::vector<std::string>> PointCloudObjectDetection::kPossibleClassNames{
    {pm::msg::ObjectClassification::CAR, {"car", "vehicle", "van", "karl_car"}},
    {pm::msg::ObjectClassification::PEDESTRIAN, {"pedestrian", "human", "man", "woman", "person", "karl_pedestrian"}},
    {pm::msg::ObjectClassification::BICYCLE, {"bicycle", "bike", "cyclist", "karl_bicycle"}},
    {pm::msg::ObjectClassification::MOTORCYCLE, {"motorcycle", "motorbike", "karl_motorcycle"}},
    {pm::msg::ObjectClassification::UTILITY, {"utility", "truck", "trailer", "train", "karl_utility"}},
    {pm::msg::ObjectClassification::BUS, {"bus", "karl_bus"}},
    {pm::msg::ObjectClassification::ANIMAL, {"animal", "karl_animal"}},
    {pm::msg::ObjectClassification::VRU, {"vru", "karl_vru"}},
    {pm::msg::ObjectClassification::MICRO, {"micro", "karl_micro"}},
    {pm::msg::ObjectClassification::ROAD_OBSTACLE, {"obstacle", "road_obstacle"}},
    {pm::msg::ObjectClassification::UNKNOWN, {"unknown"}}};

PointCloudObjectDetection::PointCloudObjectDetection(const rclcpp::NodeOptions& options)
    : rclcpp::Node("point_cloud_object_detection", options) {
  loadBootstrapParameters();
  loadManifestBackedParameterDefaults();
  declareParameters();
  loadParameters();

  // run setup after constructor has finished to enable shared_from_this()
  setup_timer_ = this->create_wall_timer(std::chrono::milliseconds(1), [this]() {
    setup();
    setup_timer_->cancel();
  });
}

void PointCloudObjectDetection::loadBootstrapParameters() {
  rcl_interfaces::msg::ParameterDescriptor repository_desc;
  repository_desc.description = "Path to the exported Triton model repository bundle root (absolute or package-relative).";
  repository_desc.additional_constraints = "Must point to a directory containing model_manifest.yml and config.pbtxt.";
  repository_desc.read_only = false;
  this->declare_parameter("prediction.model_repository", params_.model_repository, repository_desc);
  params_.model_repository = this->get_parameter("prediction.model_repository").as_string();

  rcl_interfaces::msg::ParameterDescriptor version_desc;
  version_desc.description =
      "Requested Triton model version directory inside prediction.model_repository. If empty at startup, the exported "
      "default from model_manifest.yml is used.";
  version_desc.read_only = false;
  this->declare_parameter("prediction.model_version", params_.model_version, version_desc);
  params_.model_version = this->get_parameter("prediction.model_version").as_string();
}

PointCloudObjectDetection::~PointCloudObjectDetection() {
  if (subscriber_) {
    subscriber_->shutdown();
  }
  subscriber_.reset();
  if (no_detection_zone_points_publisher_) {
    no_detection_zone_points_publisher_->shutdown();
  }
  no_detection_zone_points_publisher_.reset();
  publisher_.reset();
  no_detection_zone_pub_.reset();
  detection_area_pub_.reset();
  model_bounds_pub_.reset();

  parameters_callback_.reset();
  tf_listener_.reset();
  tf_buffer_.reset();
  detection_model_.reset();
  triton_interface_.reset();
  setup_timer_.reset();
}

template <typename T>
void PointCloudObjectDetection::declareAndLoadParameter(const std::string& name,
                                                        T& param,
                                                        const std::string& description,
                                                        const bool add_to_auto_reconfigurable_params,
                                                        const bool is_required,
                                                        const bool read_only,
                                                        const std::optional<double>& from_value,
                                                        const std::optional<double>& to_value,
                                                        const std::optional<double>& step_value,
                                                        const std::string& additional_constraints) {
  rcl_interfaces::msg::ParameterDescriptor param_desc;
  param_desc.description = description;
  param_desc.additional_constraints = additional_constraints;
  param_desc.read_only = read_only;

  auto type = rclcpp::ParameterValue(param).get_type();

  if (from_value.has_value() && to_value.has_value()) {
    if constexpr (std::is_integral_v<T>) {
      rcl_interfaces::msg::IntegerRange range;
      range.set__from_value(static_cast<T>(from_value.value())).set__to_value(static_cast<T>(to_value.value()));
      if (step_value.has_value()) range.set__step(static_cast<T>(step_value.value()));
      param_desc.integer_range = {range};
    } else if constexpr (std::is_floating_point_v<T>) {
      rcl_interfaces::msg::FloatingPointRange range;
      range.set__from_value(static_cast<T>(from_value.value())).set__to_value(static_cast<T>(to_value.value()));
      if (step_value.has_value()) range.set__step(static_cast<T>(step_value.value()));
      param_desc.floating_point_range = {range};
    } else {
      RCLCPP_WARN(this->get_logger(), "Parameter type of parameter '%s' does not support specifying a range", name.c_str());
    }
  }

  this->declare_parameter(name, type, param_desc);

  try {
    param = this->get_parameter(name).get_value<T>();
    std::stringstream ss;
    ss << "Loaded parameter '" << name << "': ";
    if constexpr (is_vector_v<T>) {
      ss << "[";
      for (const auto& element : param) ss << element << (&element != &param.back() ? ", " : "");
      ss << "]";
    } else {
      ss << param;
    }
    RCLCPP_INFO_STREAM(this->get_logger(), ss.str());
  } catch (rclcpp::exceptions::ParameterUninitializedException&) {
    if (is_required) {
      const std::string message = "Missing required parameter '" + name + "'";
      RCLCPP_FATAL_STREAM(this->get_logger(), message);
      throw std::runtime_error(message);
    } else {
      std::stringstream ss;
      ss << "Missing parameter '" << name << "', using default value: ";
      if constexpr (is_vector_v<T>) {
        ss << "[";
        for (const auto& element : param) ss << element << (&element != &param.back() ? ", " : "");
        ss << "]";
      } else {
        ss << param;
      }
      RCLCPP_WARN_STREAM(this->get_logger(), ss.str());
      this->set_parameters({rclcpp::Parameter(name, rclcpp::ParameterValue(param))});
    }
  }

  if (add_to_auto_reconfigurable_params) {
    std::function<void(const rclcpp::Parameter&)> setter = [&param](const rclcpp::Parameter& p) { param = p.get_value<T>(); };
    auto_reconfigurable_params_.push_back(std::make_tuple(name, setter));
  }
}

void PointCloudObjectDetection::declareParameters() {
  const double cscu = pm::object_access::CONTINUOUS_STATE_COVARIANCE_UNKNOWN;

  // clang-format off
  this->declareAndLoadParameter("preprocessing.backend", params_.preprocessing_backend,                                 // name
                                "Point preprocessing backend: 'cpu' or 'cuda'. If 'cuda' is selected, the node"
                                " fails fast when CUDA preprocessing support is unavailable.",                          // description
                                true,                                                                                   // add_to_auto_reconfigurable_params
                                false,                                                                                  // is_required
                                false,                                                                                  // read_only
                                std::nullopt, std::nullopt, std::nullopt,                                               // from_value, to_value, step_value
                                "Must be one of: 'cpu', 'cuda'.");                                                      // additional_constraints
  this->declareAndLoadParameter("prediction.server_url", params_.server_url,                                            // name
                                "URL of the triton server, e.g. 134.130.20.221:8001",                                   // description
                                false,                                                                                  // add_to_auto_reconfigurable_params
                                true,                                                                                   // is_required
                                true,                                                                                   // read_only
                                std::nullopt, std::nullopt, std::nullopt,                                               // from_value, to_value, step_value
                                "Must be set.");                                                                        // additional_constraints
  this->declareAndLoadParameter("prediction.triton_client_timeout_s", params_.triton_client_timeout_s,                  // name
                                "Client timeout for Triton requests in seconds (0.0 disables timeout)",                 // description
                                true,                                                                                   // add_to_auto_reconfigurable_params
                                false,                                                                                  // is_required
                                false,                                                                                  // read_only
                                0.0, kMaxTritonClientTimeoutS, std::nullopt,                                            // from_value, to_value, step_value
                                "Must be non-negative.");                                                               // additional_constraints
  this->declareAndLoadParameter("prediction.use_shm", params_.use_shm,                                                  // name
                                "Whether or not to use shared memory for Triton",                                       // description
                                true,                                                                                   // add_to_auto_reconfigurable_params
                                false,                                                                                  // is_required
                                false,                                                                                  // read_only
                                std::nullopt, std::nullopt, std::nullopt,                                               // from_value, to_value, step_value
                                "");                                                                                    // additional_constraints
  this->declareAndLoadParameter("prediction.cuda_input_shm", params_.cuda_input_shm,                                    // name
                                "If true, require Triton input tensors to use CUDA shared memory."
                                " This is only used when preprocessing.backend='cuda' and is independent of"
                                " prediction.use_shm. If CUDA shared memory is unavailable, node startup fails.",       // description
                                true,                                                                                   // add_to_auto_reconfigurable_params
                                false,                                                                                  // is_required
                                false,                                                                                  // read_only
                                std::nullopt, std::nullopt, std::nullopt,                                               // from_value, to_value, step_value
                                "");                                                                                    // additional_constraints
  this->declareAndLoadParameter("preprocessing.inference_frame", params_.inference_frame,                               // name
                                "Frame for inference",                                                                  // description
                                true,                                                                                   // add_to_auto_reconfigurable_params
                                true,                                                                                   // is_required
                                false,                                                                                  // read_only
                                std::nullopt, std::nullopt, std::nullopt,                                               // from_value, to_value, step_value
                                "Must be set.");                                                                        // additional_constraints
  this->declareAndLoadParameter("output.frame", params_.output_frame,                                                   // name
                                "Frame for object list",                                                                // description
                                true,                                                                                   // add_to_auto_reconfigurable_params
                                true,                                                                                   // is_required
                                false,                                                                                  // read_only
                                std::nullopt, std::nullopt, std::nullopt,                                               // from_value, to_value, step_value
                                "Must be set.");                                                                        // additional_constraints
  this->declareAndLoadParameter("output.sensor_id", params_.sensor_id,                                                  // name
                                "Sensor ID for object list",                                                            // description
                                true,                                                                                   // add_to_auto_reconfigurable_params
                                false,                                                                                  // is_required
                                false,                                                                                  // read_only
                                static_cast<double>(kMinSensorId), static_cast<double>(kMaxSensorId),
                                static_cast<double>(kSensorIdStep),                                                     // from_value, to_value, step_value
                                "Must be in range [" + std::to_string(kMinSensorId) + ", " +
                                    std::to_string(kMaxSensorId) + "].");                                               // additional_constraints

  this->declareAndLoadParameter(
      "output.variances", params_.variance,                                                                             // name
      "Array with variances. Entries correspond to ISCACTR model defined in perception interfaces",                     // description
      true,                                                                                                             // add_to_auto_reconfigurable_params
      false,                                                                                                            // is_required
      false,                                                                                                            // read_only
      std::nullopt, std::nullopt, std::nullopt,                                                                         // from_value, to_value, step_value
      "");                                                                                                              // additional_constraints
  sanitizeVarianceVector(params_.variance, cscu);

  this->declareAndLoadParameter("postprocessing.class_score_threshold",                                                 // name
                                params_.output_class_score_threshold,
                                "Output class score threshold. Defaults to runtime_defaults.postprocessing."
                                "class_score_threshold from the model manifest.",                                       // description
                                true,                                                                                   // add_to_auto_reconfigurable_params
                                false,                                                                                  // is_required
                                false,                                                                                  // read_only
                                kMinClassScoreThreshold, kMaxClassScoreThreshold, std::nullopt,                         // from_value, to_value, step_value
                                "Must be within [0.0, 1.0].");                                                          // additional_constraints
  this->declareAndLoadParameter("postprocessing.nms.iou_threshold", params_.nms_iou_threshold,                          // name
                                "NMS IoU threshold. Defaults to runtime_defaults.postprocessing.nms."
                                "iou_threshold from the model manifest.",                                               // description
                                true,                                                                                   // add_to_auto_reconfigurable_params
                                false,                                                                                  // is_required
                                false,                                                                                  // read_only
                                kMinClassScoreThreshold, kMaxClassScoreThreshold, std::nullopt,                         // from_value, to_value, step_value
                                "Must be within [0.0, 1.0].");                                                          // additional_constraints
  this->declareAndLoadParameter("postprocessing.nms.max_num_objects", params_.nms_max_num_objects,                      // name
                                "Maximum number of objects after NMS. Defaults to runtime_defaults.postprocessing"
                                ".nms.max_num_objects from the model manifest.",                                        // description
                                true,                                                                                   // add_to_auto_reconfigurable_params
                                false,                                                                                  // is_required
                                false,                                                                                  // read_only
                                0.0, static_cast<double>(std::numeric_limits<int32_t>::max()), std::nullopt,            // from_value, to_value, step_value
                                "Must be zero or positive.");                                                           // additional_constraints
  this->declareAndLoadParameter("postprocessing.nms.score_threshold",                                                   // name
                                params_.nms_score_threshold,
                                "NMS score threshold (single value or per-class list). Defaults to runtime_defaults"
                                ".postprocessing.nms.score_threshold from the model manifest.",                         // description
                                true,                                                                                   // add_to_auto_reconfigurable_params
                                false,                                                                                  // is_required
                                false,                                                                                  // read_only
                                std::nullopt, std::nullopt, std::nullopt,                                               // from_value, to_value, step_value
                                "Must contain exactly one value or one value per predicted class; entries must be"
                                " in [0.0, 1.0].");                                                                     // additional_constraints
  this->declareAndLoadParameter("input.point_feature_field", params_.point_feature_field,                               // name
                                "Single-feature source: 'intensity' or 'reflectivity'",                                 // description
                                true,                                                                                   // add_to_auto_reconfigurable_params
                                false,                                                                                  // is_required
                                false,                                                                                  // read_only
                                std::nullopt, std::nullopt, std::nullopt,                                               // from_value, to_value, step_value
                                "");                                                                                    // additional_constraints
  this->declareAndLoadParameter("preprocessing.point_feature.value_threshold",
                                params_.point_feature_value_threshold,                                                  // name
                                "Point-feature value threshold. Defaults to runtime_defaults.preprocessing"
                                ".point_feature.value_threshold from the model manifest.",                              // description
                                true,                                                                                   // add_to_auto_reconfigurable_params
                                false,                                                                                  // is_required
                                false,                                                                                  // read_only
                                0.0, 1000000.0, std::nullopt,                                                           // from_value, to_value, step_value
                                "Must be greater than 0 when value_threshold normalization is used.");                  // additional_constraints
  this->declareAndLoadParameter("preprocessing.detection_area.z_min", params_.detection_area_z_min,                     // name
                                "Effective preprocessing lower z-bound used for point filtering and tensor "
                                "construction. Runtime override; defaults to the model manifest z range. Values"
                                " outside the manifest z range are accepted with a warning.",                         // description
                                true,                                                                                   // add_to_auto_reconfigurable_params
                                false,                                                                                  // is_required
                                false,                                                                                  // read_only
                                std::nullopt, std::nullopt, std::nullopt,                                               // from_value, to_value, step_value
                                "Must be finite and smaller than preprocessing.detection_area.z_max.");                 // additional_constraints
  this->declareAndLoadParameter("preprocessing.detection_area.z_max", params_.detection_area_z_max,                     // name
                                "Effective preprocessing upper z-bound used for point filtering and tensor "
                                "construction. Runtime override; defaults to the model manifest z range. Values"
                                " outside the manifest z range are accepted with a warning.",                         // description
                                true,                                                                                   // add_to_auto_reconfigurable_params
                                false,                                                                                  // is_required
                                false,                                                                                  // read_only
                                std::nullopt, std::nullopt, std::nullopt,                                               // from_value, to_value, step_value
                                "Must be finite and greater than preprocessing.detection_area.z_min.");                 // additional_constraints

  this->declareAndLoadParameter("preprocessing.no_detection_zone.enabled", params_.no_detection_zone_enabled,
                                "Enable rectangular no-detection zone in inference_frame",                              // description
                                true,                                                                                   // add_to_auto_reconfigurable_params
                                false,                                                                                  // is_required
                                false,                                                                                  // read_only
                                std::nullopt, std::nullopt, std::nullopt,                                               // from_value, to_value, step_value
                                "");                                                                                    // additional_constraints
  this->declareAndLoadParameter("preprocessing.no_detection_zone.remove_points", params_.no_detection_zone_remove_points,
                                "If true, remove raw points inside the no-detection zone from model input and"
                                " unclassified point publishing",                                                       // description
                                true,                                                                                   // add_to_auto_reconfigurable_params
                                false,                                                                                  // is_required
                                false,                                                                                  // read_only
                                std::nullopt, std::nullopt, std::nullopt,                                               // from_value, to_value, step_value
                                "");                                                                                    // additional_constraints
  this->declareAndLoadParameter("preprocessing.no_detection_zone.x_min", params_.no_detection_zone_x_min,               // name
                                "No-detection zone x_min (inference_frame)",                                            // description
                                true,                                                                                   // add_to_auto_reconfigurable_params
                                false,                                                                                  // is_required
                                false,                                                                                  // read_only
                                std::nullopt, std::nullopt, std::nullopt,                                               // from_value, to_value, step_value
                                "");                                                                                    // additional_constraints
  this->declareAndLoadParameter("preprocessing.no_detection_zone.x_max", params_.no_detection_zone_x_max,               // name
                                "No-detection zone x_max (inference_frame)",                                            // description
                                true,                                                                                   // add_to_auto_reconfigurable_params
                                false,                                                                                  // is_required
                                false,                                                                                  // read_only
                                std::nullopt, std::nullopt, std::nullopt,                                               // from_value, to_value, step_value
                                "");                                                                                    // additional_constraints
  this->declareAndLoadParameter("preprocessing.no_detection_zone.y_min", params_.no_detection_zone_y_min,               // name
                                "No-detection zone y_min (inference_frame)",                                            // description
                                true,                                                                                   // add_to_auto_reconfigurable_params
                                false,                                                                                  // is_required
                                false,                                                                                  // read_only
                                std::nullopt, std::nullopt, std::nullopt,                                               // from_value, to_value, step_value
                                "");                                                                                    // additional_constraints
  this->declareAndLoadParameter("preprocessing.no_detection_zone.y_max", params_.no_detection_zone_y_max,               // name
                                "No-detection zone y_max (inference_frame)",                                            // description
                                true,                                                                                   // add_to_auto_reconfigurable_params
                                false,                                                                                  // is_required
                                false,                                                                                  // read_only
                                std::nullopt, std::nullopt, std::nullopt,                                               // from_value, to_value, step_value
                                "");                                                                                    // additional_constraints
  this->declareAndLoadParameter("preprocessing.no_detection_zone.publish_polygon", params_.no_detection_zone_publish_polygon,
                                "If true, publish a geometry_msgs/PolygonStamped with the no-detection zone bounds",    // description
                                true,                                                                                   // add_to_auto_reconfigurable_params
                                false,                                                                                  // is_required
                                false,                                                                                  // read_only
                                std::nullopt, std::nullopt, std::nullopt,                                               // from_value, to_value, step_value
                                "");                                                                                    // additional_constraints
  this->declareAndLoadParameter("preprocessing.no_detection_zone.publish_points", params_.no_detection_zone_publish_points,
                                "If true, publish raw points inside the no-detection zone",                             // description
                                true,                                                                                   // add_to_auto_reconfigurable_params
                                false,                                                                                  // is_required
                                false,                                                                                  // read_only
                                std::nullopt, std::nullopt, std::nullopt,                                               // from_value, to_value, step_value
                                "");                                                                                    // additional_constraints

  this->declareAndLoadParameter("preprocessing.detection_area.enabled", params_.detection_area_enabled,                 // name
                                "Enable circular-sector detection area",                                                // description
                                true,                                                                                   // add_to_auto_reconfigurable_params
                                false,                                                                                  // is_required
                                false,                                                                                  // read_only
                                std::nullopt, std::nullopt, std::nullopt,                                               // from_value, to_value, step_value
                                "");                                                                                    // additional_constraints
  this->declareAndLoadParameter("preprocessing.detection_area.center_x", params_.detection_area_center_x,               // name
                                "Detection area center x (m) in inference_frame",                                       // description
                                true,                                                                                   // add_to_auto_reconfigurable_params
                                false,                                                                                  // is_required
                                false,                                                                                  // read_only
                                std::nullopt, std::nullopt, std::nullopt,                                               // from_value, to_value, step_value
                                "");                                                                                    // additional_constraints
  this->declareAndLoadParameter("preprocessing.detection_area.center_y", params_.detection_area_center_y,               // name
                                "Detection area center y (m) in inference_frame",                                       // description
                                true,                                                                                   // add_to_auto_reconfigurable_params
                                false,                                                                                  // is_required
                                false,                                                                                  // read_only
                                std::nullopt, std::nullopt, std::nullopt,                                               // from_value, to_value, step_value
                                "");                                                                                    // additional_constraints
  this->declareAndLoadParameter("preprocessing.detection_area.radius", params_.detection_area_radius,                   // name
                                "Detection area radius (m)",                                                            // description
                                true,                                                                                   // add_to_auto_reconfigurable_params
                                false,                                                                                  // is_required
                                false,                                                                                  // read_only
                                kMinDetectionAreaRadius, kMaxDetectionAreaRadius, std::nullopt,                         // from_value, to_value, step_value
                                "Must be non-negative.");                                                               // additional_constraints
  this->declareAndLoadParameter("preprocessing.detection_area.bearing_deg", params_.detection_area_bearing_deg,
                                "Detection area central azimuth (deg, 0 along +x, CCW positive)",                       // description
                                true,                                                                                   // add_to_auto_reconfigurable_params
                                false,                                                                                  // is_required
                                false,                                                                                  // read_only
                                kMinDetectionAreaBearingDeg, kMaxDetectionAreaBearingDeg, std::nullopt,                 // from_value, to_value, step_value
                                "Must be within [-360, 360].");                                                         // additional_constraints
  this->declareAndLoadParameter("preprocessing.detection_area.fov_deg", params_.detection_area_fov_deg,                 // name
                                "Detection area FOV angle (deg)",                                                       // description
                                true,                                                                                   // add_to_auto_reconfigurable_params
                                false,                                                                                  // is_required
                                false,                                                                                  // read_only
                                kMinDetectionAreaFovDeg, kMaxDetectionAreaFovDeg, std::nullopt,                         // from_value, to_value, step_value
                                "Must be in the range (0, 360].");                                                      // additional_constraints
  this->declareAndLoadParameter("preprocessing.detection_area.publish_polygon", params_.detection_area_publish_polygon,
                                "Publish geometry_msgs/PolygonStamped approximating the sector",                        // description
                                true,                                                                                   // add_to_auto_reconfigurable_params
                                false,                                                                                  // is_required
                                false,                                                                                  // read_only
                                std::nullopt, std::nullopt, std::nullopt,                                               // from_value, to_value, step_value
                                "");                                                                                    // additional_constraints
  this->declareAndLoadParameter("preprocessing.detection_area.num_segments", params_.detection_area_num_segments,
                                "Number of segments to approximate the circular arc (>= 3)",                            // description
                                true,                                                                                   // add_to_auto_reconfigurable_params
                                false,                                                                                  // is_required
                                false,                                                                                  // read_only
                                static_cast<double>(kMinDetectionAreaNumSegments),
                                static_cast<double>(kMaxDetectionAreaNumSegments),
                                std::nullopt,                                                                           // from_value, to_value, step_value
                                "Must be greater than or equal to 3.");                                                 // additional_constraints
  this->declareAndLoadParameter("preprocessing.detection_area.filter_detections", params_.detection_area_filter_detections,
                                "Remove detections outside the detection area",                                         // description
                                true,                                                                                   // add_to_auto_reconfigurable_params
                                false,                                                                                  // is_required
                                false,                                                                                  // read_only
                                std::nullopt, std::nullopt, std::nullopt,                                               // from_value, to_value, step_value
                                "");                                                                                    // additional_constraints
  this->declareAndLoadParameter("preprocessing.detection_area.filter_mode", params_.detection_area_filter_mode,
                                "Filtering mode: 'center' or 'complete'",                                               // description
                                true,                                                                                   // add_to_auto_reconfigurable_params
                                false,                                                                                  // is_required
                                false,                                                                                  // read_only
                                std::nullopt, std::nullopt, std::nullopt,                                               // from_value, to_value, step_value
                                "Must be one of: 'center', 'complete'.");                                               // additional_constraints

  this->declareAndLoadParameter("output.model_bounds.publish_polygon", params_.model_bounds_publish_polygon,
                                "Publish the model x/y range rectangle as geometry_msgs/PolygonStamped",                // description
                                true,                                                                                   // add_to_auto_reconfigurable_params
                                false,                                                                                  // is_required
                                false,                                                                                  // read_only
                                std::nullopt, std::nullopt, std::nullopt,                                               // from_value, to_value, step_value
                                "");                                                                                    // additional_constraints
  this->declareAndLoadParameter("output.grid_maps.frame", params_.grid_map_frame,
                                "Frame for auxiliary grid-map publication. If empty, use the inference frame.",         // description
                                true,                                                                                   // add_to_auto_reconfigurable_params
                                false,                                                                                  // is_required
                                false,                                                                                  // read_only
                                std::nullopt, std::nullopt, std::nullopt,                                               // from_value, to_value, step_value
                                "");                                                                                    // additional_constraints
  this->declareAndLoadParameter("output.grid_maps.publish_density", params_.publish_density_grid_map,
                                "Publish decoded density logits as an auxiliary grid map",                              // description
                                true,                                                                                   // add_to_auto_reconfigurable_params
                                false,                                                                                  // is_required
                                false,                                                                                  // read_only
                                std::nullopt, std::nullopt, std::nullopt,                                               // from_value, to_value, step_value
                                "");                                                                                    // additional_constraints
  this->declareAndLoadParameter("output.grid_maps.publish_occupancy", params_.publish_occupancy_grid_map,
                                "Publish decoded occupancy logits as an auxiliary grid map",                            // description
                                true,                                                                                   // add_to_auto_reconfigurable_params
                                false,                                                                                  // is_required
                                false,                                                                                  // read_only
                                std::nullopt, std::nullopt, std::nullopt,                                               // from_value, to_value, step_value
                                "");                                                                                    // additional_constraints
  this->declareAndLoadParameter("output.grid_maps.publish_combined", params_.publish_combined_grid_map,
                                "Publish a combined auxiliary occupancy grid map",                                      // description
                                true,                                                                                   // add_to_auto_reconfigurable_params
                                false,                                                                                  // is_required
                                false,                                                                                  // read_only
                                std::nullopt, std::nullopt, std::nullopt,                                               // from_value, to_value, step_value
                                "");                                                                                    // additional_constraints
  this->declareAndLoadParameter("output.grid_maps.publish_static", params_.publish_static_grid_map,
                                "Publish a static-obstacle occupancy grid map",                                         // description
                                true,                                                                                   // add_to_auto_reconfigurable_params
                                false,                                                                                  // is_required
                                false,                                                                                  // read_only
                                std::nullopt, std::nullopt, std::nullopt,                                               // from_value, to_value, step_value
                                "");                                                                                    // additional_constraints
  this->declareAndLoadParameter("output.grid_maps.zero_in_no_detection_zone",
                                params_.zero_grid_map_cells_in_no_detection_zone,
                                "If true, set published auxiliary grid-map cells inside the configured no-detection"
                                " zone to zero",                                                                        // description
                                true,                                                                                   // add_to_auto_reconfigurable_params
                                false,                                                                                  // is_required
                                false,                                                                                  // read_only
                                std::nullopt, std::nullopt, std::nullopt,                                               // from_value, to_value, step_value
                                "");                                                                                    // additional_constraints
  this->declareAndLoadParameter("output.grid_maps.zero_outside_detection_area",
                                params_.zero_grid_map_cells_outside_detection_area,
                                "If true, set published auxiliary grid-map cells outside the configured detection"
                                " area to zero",                                                                        // description
                                true,                                                                                   // add_to_auto_reconfigurable_params
                                false,                                                                                  // is_required
                                false,                                                                                  // read_only
                                std::nullopt, std::nullopt, std::nullopt,                                               // from_value, to_value, step_value
                                "");                                                                                    // additional_constraints
  this->declareAndLoadParameter("output.grid_maps.density_gain", params_.density_grid_map_gain,
                                "Linear gain applied to the published density grid map",                                // description
                                true,                                                                                   // add_to_auto_reconfigurable_params
                                false,                                                                                  // is_required
                                false,                                                                                  // read_only
                                0.0, 100.0, std::nullopt,                                                               // from_value, to_value, step_value
                                "Must be within [0, 100].");                                                            // additional_constraints
  this->declareAndLoadParameter("output.grid_maps.occupancy_gain", params_.occupancy_grid_map_gain,
                                "Linear gain applied to the published occupancy grid map",                              // description
                                true,                                                                                   // add_to_auto_reconfigurable_params
                                false,                                                                                  // is_required
                                false,                                                                                  // read_only
                                0.0, 100.0, std::nullopt,                                                               // from_value, to_value, step_value
                                "Must be within [0, 100].");                                                            // additional_constraints
  this->declareAndLoadParameter("output.grid_maps.combined_gain", params_.combined_grid_map_gain,
                                "Linear gain applied to the published combined grid map",                               // description
                                true,                                                                                   // add_to_auto_reconfigurable_params
                                false,                                                                                  // is_required
                                false,                                                                                  // read_only
                                0.0, 100.0, std::nullopt,                                                               // from_value, to_value, step_value
                                "Must be within [0, 100].");                                                            // additional_constraints
  this->declareAndLoadParameter("output.grid_maps.static_gain", params_.static_grid_map_gain,
                                "Linear gain applied to the published static grid map",                                 // description
                                true,                                                                                   // add_to_auto_reconfigurable_params
                                false,                                                                                  // is_required
                                false,                                                                                  // read_only
                                0.0, 100.0, std::nullopt,                                                               // from_value, to_value, step_value
                                "Must be within [0, 100].");                                                            // additional_constraints

  validateParamsOrThrow();
  // clang-format on
}

void PointCloudObjectDetection::syncModelRuntimeConfigFromParams() { syncModelRuntimeConfigFromParams(model_config_, params_); }

void PointCloudObjectDetection::syncModelRuntimeConfigFromParams(ModelConfig& model_config, const Params& params) const {
  model_config.preprocessing_backend = params.preprocessing_backend;
  model_config.point_feature_value_threshold = static_cast<float>(params.point_feature_value_threshold);
  model_config.z_min = static_cast<float>(params.detection_area_z_min);
  model_config.z_max = static_cast<float>(params.detection_area_z_max);
  if (model_config.pillar_map_range.size() >= 3 && model_config.pillar_map_range[2].size() >= 2) {
    model_config.pillar_map_range[2][0] = model_config.z_min;
    model_config.pillar_map_range[2][1] = model_config.z_max;
  }
  model_config.no_detection_zone_remove_points = params.no_detection_zone_remove_points;
  model_config.no_detection_zone_x_min = params.no_detection_zone_x_min;
  model_config.no_detection_zone_x_max = params.no_detection_zone_x_max;
  model_config.no_detection_zone_y_min = params.no_detection_zone_y_min;
  model_config.no_detection_zone_y_max = params.no_detection_zone_y_max;
  model_config.detection_area_center_x = params.detection_area_center_x;
  model_config.detection_area_center_y = params.detection_area_center_y;
  model_config.detection_area_radius = params.detection_area_radius;
  model_config.detection_area_bearing_deg = params.detection_area_bearing_deg;
  model_config.detection_area_fov_deg = params.detection_area_fov_deg;
  model_config.detection_area_remove_points_outside = params.detection_area_enabled &&
                                                      params.detection_area_radius > kMinDetectionAreaRadius &&
                                                      params.detection_area_fov_deg > kMinDetectionAreaFovDeg &&
                                                      params.detection_area_num_segments >= kMinDetectionAreaNumSegments;
}

void PointCloudObjectDetection::syncNmsRuntimeConfigFromParams() {
  syncNmsRuntimeConfigFromParams(model_config_, nms_config_, params_);
}

void PointCloudObjectDetection::syncNmsRuntimeConfigFromParams(ModelConfig& model_config,
                                                               pcod_common::NmsConfig& nms_config,
                                                               const Params& params) const {
  std::vector<double> score_thresholds = params.nms_score_threshold;
  if (!score_thresholds.empty()) {
    if (!updateNMSScoreThreshold(score_thresholds, model_config.predicted_class_names)) {
      throwParameterError(this->get_logger(), "postprocessing.nms.score_threshold",
                          "must contain exactly one value or one per predicted class");
    }
  }
  model_config.nms_score_threshold = std::move(score_thresholds);
  model_config.nms_iou_threshold = static_cast<float>(params.nms_iou_threshold);
  model_config.nms_max_num_objects = static_cast<int>(params.nms_max_num_objects);

  nms_config.iou_threshold = static_cast<float>(model_config.nms_iou_threshold);
  nms_config.max_detections = model_config.nms_max_num_objects;
  nms_config.score_thresholds.clear();
  for (double thresh : model_config.nms_score_threshold) {
    nms_config.score_thresholds.push_back(static_cast<float>(thresh));
  }
}

void PointCloudObjectDetection::loadParameters() { syncModelRuntimeConfigFromParams(); }

void PointCloudObjectDetection::loadManifestBackedParameterDefaults() {
  auto fail = [this](const std::string& param, const std::string& message) {
    throwParameterError(this->get_logger(), param, message);
  };

  if (params_.model_repository.empty()) {
    fail("prediction.model_repository", "must be set to the exported Triton model repository path");
  }

  const std::string repository_path = resolveModelRepositoryPath(params_.model_repository);
  if (!std::filesystem::exists(repository_path) || !std::filesystem::is_directory(repository_path)) {
    fail("prediction.model_repository", "must point to an existing directory");
  }

  const std::string manifest_path = manifestPathFromRepository(repository_path);
  if (!std::filesystem::exists(manifest_path)) {
    fail("prediction.model_repository", "model_manifest.yml does not exist inside the configured repository");
  }

  const pcod_common::ModelManifest manifest = pcod_common::LoadModelManifest(manifest_path);
  pcod_common::ValidateModelManifest(manifest);

  if (!manifest.artifact.triton.enabled) {
    fail("prediction.model_repository", "repository manifest must describe a Triton export");
  }
  const std::filesystem::path config_path = std::filesystem::path(repository_path) / manifest.artifact.files.triton_config;
  if (manifest.artifact.files.triton_config.empty() || !std::filesystem::exists(config_path)) {
    fail("prediction.model_repository", "repository manifest must reference an existing config.pbtxt");
  }

  try {
    params_.model_name = parseTritonModelNameFromConfig(config_path.string());
  } catch (const std::exception& e) {
    fail("prediction.model_repository", std::string("failed to parse config.pbtxt: ") + e.what());
  }
  if (params_.model_name != manifest.artifact.triton.model_name) {
    fail("prediction.model_repository",
         "config.pbtxt model name does not match artifact.triton.model_name in model_manifest.yml");
  }

  if (params_.model_version.empty()) {
    params_.model_version = manifest.artifact.triton.model_version;
  } else if (params_.model_version != manifest.artifact.triton.model_version) {
    RCLCPP_WARN(this->get_logger(),
                "prediction.model_version='%s' differs from manifest default '%s'; using the requested version.",
                params_.model_version.c_str(), manifest.artifact.triton.model_version.c_str());
  }
  params_.point_feature_value_threshold = manifest.runtime_defaults.preprocessing.point_feature.value_threshold;
  params_.detection_area_z_min = manifest.frozen_contract.preprocessing.z_range[0];
  params_.detection_area_z_max = manifest.frozen_contract.preprocessing.z_range[1];
  params_.output_class_score_threshold = manifest.runtime_defaults.postprocessing.class_score_threshold;
  params_.nms_iou_threshold = manifest.runtime_defaults.postprocessing.nms_iou_threshold;
  params_.nms_max_num_objects = manifest.runtime_defaults.postprocessing.max_detections;
  params_.nms_score_threshold.assign(manifest.runtime_defaults.postprocessing.nms_score_thresholds.begin(),
                                     manifest.runtime_defaults.postprocessing.nms_score_thresholds.end());
}

void PointCloudObjectDetection::validateParamsOrThrow() const {
  auto fail = [this](const std::string& param, const std::string& message) {
    throwParameterError(this->get_logger(), param, message);
  };

  const double cscu = pm::object_access::CONTINUOUS_STATE_COVARIANCE_UNKNOWN;
  if (params_.variance.size() != kExpectedVarianceSize) {
    fail("output.variances",
         "must contain " + std::to_string(kExpectedVarianceSize) + " entries corresponding to the ISCACTR state");
  }
  for (std::size_t i = 0; i < params_.variance.size(); ++i) {
    const double value = params_.variance[i];
    if (!isFinite(value)) {
      fail("output.variances", "all entries must be finite");
    }
    if (value < 0.0 && std::fabs(value - cscu) > 1e-9) {
      fail("output.variances", "entries must be non-negative or equal to CONTINUOUS_STATE_COVARIANCE_UNKNOWN (-1)");
    }
  }

  if (!isAllowedPointFeatureField(params_.point_feature_field)) {
    fail("input.point_feature_field", "must be one of: " + allowedPointFeatureFieldsString());
  }
  if (!isAllowedPreprocessingBackend(params_.preprocessing_backend)) {
    fail("preprocessing.backend", "must be one of: " + allowedPreprocessingBackendsString());
  }
  if (!isFinite(params_.density_grid_map_gain) || params_.density_grid_map_gain < 0.0 || params_.density_grid_map_gain > 100.0) {
    fail("output.grid_maps.density_gain", "must be finite and within [0, 100]");
  }
  if (!isFinite(params_.occupancy_grid_map_gain) || params_.occupancy_grid_map_gain < 0.0 ||
      params_.occupancy_grid_map_gain > 100.0) {
    fail("output.grid_maps.occupancy_gain", "must be finite and within [0, 100]");
  }
  if (!isFinite(params_.combined_grid_map_gain) || params_.combined_grid_map_gain < 0.0 ||
      params_.combined_grid_map_gain > 100.0) {
    fail("output.grid_maps.combined_gain", "must be finite and within [0, 100]");
  }
  if (!isFinite(params_.static_grid_map_gain) || params_.static_grid_map_gain < 0.0 || params_.static_grid_map_gain > 100.0) {
    fail("output.grid_maps.static_gain", "must be finite and within [0, 100]");
  }
  if (!isFinite(params_.point_feature_value_threshold)) {
    fail("preprocessing.point_feature.value_threshold", "must be finite");
  }
  if (!isFinite(params_.detection_area_z_min)) {
    fail("preprocessing.detection_area.z_min", "must be finite");
  }
  if (!isFinite(params_.detection_area_z_max)) {
    fail("preprocessing.detection_area.z_max", "must be finite");
  }
  if (!(params_.detection_area_z_min < params_.detection_area_z_max)) {
    fail("preprocessing.detection_area.z_min", "must be smaller than preprocessing.detection_area.z_max");
  }
  if (!isFinite(params_.output_class_score_threshold)) {
    fail("postprocessing.class_score_threshold", "must be finite");
  }
  if (params_.output_class_score_threshold < kMinClassScoreThreshold ||
      params_.output_class_score_threshold > kMaxClassScoreThreshold) {
    fail("postprocessing.class_score_threshold", "must be within [0.0, 1.0]");
  }
  for (double threshold : params_.nms_score_threshold) {
    if (!isProbability(threshold)) {
      fail("postprocessing.nms.score_threshold", "each entry must be within [0.0, 1.0]");
    }
  }
  if (!isFinite(params_.nms_iou_threshold)) {
    fail("postprocessing.nms.iou_threshold", "must be finite");
  }
  if (params_.nms_iou_threshold < kMinClassScoreThreshold || params_.nms_iou_threshold > kMaxClassScoreThreshold) {
    fail("postprocessing.nms.iou_threshold", "must be within [0.0, 1.0]");
  }
  if (params_.nms_max_num_objects < 0) {
    fail("postprocessing.nms.max_num_objects", "must be zero or positive");
  }
  if (params_.server_url.empty()) {
    fail("prediction.server_url", "must be set to the Triton server address");
  }
  if (!isFinite(params_.triton_client_timeout_s)) {
    fail("prediction.triton_client_timeout_s", "must be finite");
  }
  if (params_.model_repository.empty()) {
    fail("prediction.model_repository", "must be set to the exported Triton model repository path");
  }
  const std::string resolved_repository_path = resolveModelRepositoryPath(params_.model_repository);
  if (!std::filesystem::exists(resolved_repository_path) || !std::filesystem::is_directory(resolved_repository_path)) {
    fail("prediction.model_repository", "must point to an existing directory");
  }
  const std::string resolved_manifest_path = manifestPathFromRepository(resolved_repository_path);
  if (!std::filesystem::exists(resolved_manifest_path)) {
    fail("prediction.model_repository", "model_manifest.yml does not exist inside the configured repository");
  }
  if (params_.model_version.empty()) {
    fail("prediction.model_version", "must not be empty after resolving repository defaults");
  }
  const std::filesystem::path version_path = std::filesystem::path(resolved_repository_path) / params_.model_version;
  if (!std::filesystem::exists(version_path) || !std::filesystem::is_directory(version_path)) {
    fail("prediction.model_version", "requested model version directory does not exist inside model_repository");
  }

  if (!isFinite(params_.no_detection_zone_x_min) || !isFinite(params_.no_detection_zone_x_max) ||
      !isFinite(params_.no_detection_zone_y_min) || !isFinite(params_.no_detection_zone_y_max)) {
    fail("preprocessing.no_detection_zone", "bounds must be finite values");
  }
  if (params_.no_detection_zone_enabled) {
    if (!(params_.no_detection_zone_x_min < params_.no_detection_zone_x_max) ||
        !(params_.no_detection_zone_y_min < params_.no_detection_zone_y_max)) {
      fail("preprocessing.no_detection_zone", "requires x_min < x_max and y_min < y_max when enabled");
    }
  }

  if (!isFinite(params_.detection_area_center_x) || !isFinite(params_.detection_area_center_y)) {
    fail("preprocessing.detection_area.center", "center_x and center_y must be finite");
  }
  if (!isFinite(params_.detection_area_radius)) {
    fail("preprocessing.detection_area.radius", "must be finite");
  }
  if (!isFinite(params_.detection_area_fov_deg)) {
    fail("preprocessing.detection_area.fov_deg", "must be finite");
  }
  if (params_.detection_area_fov_deg <= kMinDetectionAreaFovDeg) {
    fail("preprocessing.detection_area.fov_deg", "must be greater than 0");
  }
  if (!isFinite(params_.detection_area_bearing_deg)) {
    fail("preprocessing.detection_area.bearing_deg", "must be finite");
  }
  if (params_.detection_area_enabled && params_.detection_area_radius <= kMinDetectionAreaRadius) {
    fail("preprocessing.detection_area.radius", "must be greater than 0 when detection_area.enabled is true");
  }
  if (params_.detection_area_filter_mode != "center" && params_.detection_area_filter_mode != "complete") {
    fail("preprocessing.detection_area.filter_mode", "must be either 'center' or 'complete'");
  }
}

bool PointCloudObjectDetection::updateNMSScoreThreshold(std::vector<double>& score_thresholds,
                                                        const std::vector<std::string>& predicted_class_names) const {
  if (score_thresholds.size() == 1) {
    for (std::size_t i = 1; i < predicted_class_names.size(); i++) {
      // If a single NMS score threshold is provided, replicate it for each class.
      score_thresholds.push_back(score_thresholds[0]);
    }
  }
  if (score_thresholds.size() != predicted_class_names.size()) {
    RCLCPP_ERROR(this->get_logger(),
                 "NMS score threshold count mismatch: got %zu value(s), expected 1 or %zu (one per predicted class).",
                 score_thresholds.size(), predicted_class_names.size());
    return false;
  }
  return true;
}

ModelConfig PointCloudObjectDetection::loadModelConfig(const Params& params,
                                                       std::string& model_name,
                                                       pcod_common::NmsConfig& nms_config) const {
  const std::string repository_path = resolveModelRepositoryPath(params.model_repository);
  const std::string manifest_path = manifestPathFromRepository(repository_path);
  pcod_common::ModelManifest manifest = pcod_common::LoadModelManifest(manifest_path);
  pcod_common::ValidateModelManifest(manifest);
  ModelConfig model_config;

  const std::string manifest_precision = boost::algorithm::to_lower_copy(manifest.artifact.precision);
  if (!isSupportedManifestPrecision(manifest_precision)) {
    throwParameterError(this->get_logger(), "prediction.model_repository",
                        "manifest precision must be one of 'fp32', 'fp16', or 'int8'");
  }
  const std::string manifest_device = boost::algorithm::to_lower_copy(manifest.artifact.device);
  if (manifest_device.rfind("cuda", 0) != 0) {
    throwParameterError(this->get_logger(), "prediction.model_repository", "manifest device must start with 'cuda'");
  }
  if (!manifest.artifact.triton.enabled || manifest.artifact.triton.model_name.empty() ||
      manifest.artifact.triton.model_version.empty()) {
    throwParameterError(this->get_logger(), "prediction.model_repository",
                        "manifest must define artifact.triton.model_name and artifact.triton.model_version");
  }
  const std::filesystem::path config_path = std::filesystem::path(repository_path) / manifest.artifact.files.triton_config;
  if (manifest.artifact.files.triton_config.empty() || !std::filesystem::exists(config_path)) {
    throwParameterError(this->get_logger(), "prediction.model_repository",
                        "manifest must reference an existing Triton config.pbtxt");
  }
  try {
    model_name = parseTritonModelNameFromConfig(config_path.string());
  } catch (const std::exception& e) {
    throwParameterError(this->get_logger(), "prediction.model_repository",
                        std::string("failed to parse config.pbtxt: ") + e.what());
  }
  if (model_name != manifest.artifact.triton.model_name) {
    throwParameterError(this->get_logger(), "prediction.model_repository",
                        "config.pbtxt model name does not match artifact.triton.model_name");
  }

  model_config.point_feature_normalization_type = manifest.frozen_contract.preprocessing.point_feature_normalization.type;
  model_config.point_feature_value_threshold = manifest.runtime_defaults.preprocessing.point_feature.value_threshold;
  model_config.point_feature_min_value = manifest.frozen_contract.preprocessing.point_feature_normalization.min_value;
  model_config.point_feature_max_value = manifest.frozen_contract.preprocessing.point_feature_normalization.max_value;
  model_config.point_feature_norm_epsilon = manifest.frozen_contract.preprocessing.point_feature_normalization.epsilon;
  model_config.predicted_class_names = manifest.frozen_contract.postprocessing.class_names;
  model_config.class_mapping_.clear();

  model_config.x_min = manifest.frozen_contract.preprocessing.x_range[0];
  model_config.x_max = manifest.frozen_contract.preprocessing.x_range[1];
  model_config.y_min = manifest.frozen_contract.preprocessing.y_range[0];
  model_config.y_max = manifest.frozen_contract.preprocessing.y_range[1];
  model_config.z_min = manifest.frozen_contract.preprocessing.z_range[0];
  model_config.z_max = manifest.frozen_contract.preprocessing.z_range[1];
  model_config.contract_z_min = manifest.frozen_contract.preprocessing.z_range[0];
  model_config.contract_z_max = manifest.frozen_contract.preprocessing.z_range[1];
  model_config.x_grid_size = manifest.frozen_contract.postprocessing.grid_x;
  model_config.y_grid_size = manifest.frozen_contract.postprocessing.grid_y;
  model_config.voxel_x = manifest.frozen_contract.preprocessing.voxel_x;
  model_config.voxel_y = manifest.frozen_contract.preprocessing.voxel_y;
  model_config.voxel_z = manifest.frozen_contract.preprocessing.voxel_z;

  model_config.nms_iou_threshold = static_cast<float>(params.nms_iou_threshold);
  model_config.nms_max_num_objects = static_cast<int>(params.nms_max_num_objects);
  model_config.nms_score_threshold.assign(params.nms_score_threshold.begin(), params.nms_score_threshold.end());

  model_config.max_num_points = manifest.frozen_contract.preprocessing.max_num_points;
  model_config.stride.clear();
  for (int stride : manifest.frozen_contract.model.stride) {
    model_config.stride.push_back(stride);
  }
  model_config.first_up_stride = manifest.frozen_contract.model.first_up_stride;

  model_config.pillar_map_size = {manifest.frozen_contract.model.pillar_map_size[0],
                                  manifest.frozen_contract.model.pillar_map_size[1]};
  model_config.pillar_map_range = {
      {manifest.frozen_contract.model.pillar_map_range[0][0], manifest.frozen_contract.model.pillar_map_range[0][1]},
      {manifest.frozen_contract.model.pillar_map_range[1][0], manifest.frozen_contract.model.pillar_map_range[1][1]},
      {manifest.frozen_contract.model.pillar_map_range[2][0], manifest.frozen_contract.model.pillar_map_range[2][1]}};

  for (auto& name : model_config.predicted_class_names) {
    boost::algorithm::to_lower(name);
    uint8_t pm_type = pm::msg::ObjectClassification::UNCLASSIFIED;
    for (auto& [type, possible_names] : kPossibleClassNames) {
      if (std::find(possible_names.begin(), possible_names.end(), name) != possible_names.end()) {
        pm_type = type;
        break;
      }
    }
    if (pm_type == pm::msg::ObjectClassification::UNCLASSIFIED) {
      RCLCPP_WARN_STREAM(get_logger(),
                         "The class "
                         " << name << "
                         " is not mapped to any class of"
                             << " perception_msgs::msg::ObjectClassification");
    }
    model_config.class_mapping_[name] = pm_type;
  }

  syncModelRuntimeConfigFromParams(model_config, params);
  syncNmsRuntimeConfigFromParams(model_config, nms_config, params);

  if (model_config.nms_max_num_objects < 0) {
    throwParameterError(this->get_logger(), "postprocessing.nms.max_num_objects", "cannot be negative");
  }
  if (!updateNMSScoreThreshold(model_config.nms_score_threshold, model_config.predicted_class_names)) {
    throwParameterError(this->get_logger(), "postprocessing.nms.score_threshold",
                        "must contain exactly one value or one per predicted class");
  }
  return model_config;
}
void PointCloudObjectDetection::validateModelConfigOrThrow() const { validateModelConfigOrThrow(model_config_); }

void PointCloudObjectDetection::validateModelConfigOrThrow(const ModelConfig& model_config) const {
  auto fail = [this](const std::string& param, const std::string& message) {
    throwParameterError(this->get_logger(), param, message);
  };

  if (!isAllowedPreprocessingBackend(model_config.preprocessing_backend)) {
    fail("preprocessing.backend", "must be one of: " + allowedPreprocessingBackendsString());
  }
  if (model_config.preprocessing_backend == "cuda" && !pcod_common::HasCudaPillarPreprocessSupport()) {
    fail("preprocessing.backend", "is set to 'cuda' but CUDA preprocessing support is not available in this build/runtime");
  }
  if (!isFinite(model_config.x_min) || !isFinite(model_config.x_max)) {
    fail("x_min/x_max", "must be finite");
  }
  if (!(model_config.x_min < model_config.x_max)) {
    fail("x_min/x_max", "requires x_min < x_max");
  }

  if (!isFinite(model_config.y_min) || !isFinite(model_config.y_max)) {
    fail("y_min/y_max", "must be finite");
  }
  if (!(model_config.y_min < model_config.y_max)) {
    fail("y_min/y_max", "requires y_min < y_max");
  }

  if (!isFinite(model_config.z_min) || !isFinite(model_config.z_max)) {
    fail("z_min/z_max", "must be finite");
  }
  if (!(model_config.z_min < model_config.z_max)) {
    fail("z_min/z_max", "requires z_min < z_max");
  }
  if (!isFinite(model_config.contract_z_min) || !isFinite(model_config.contract_z_max)) {
    fail("contract_z_min/contract_z_max", "must be finite");
  }
  if (!(model_config.contract_z_min < model_config.contract_z_max)) {
    fail("contract_z_min/contract_z_max", "requires contract_z_min < contract_z_max");
  }
  if (model_config.z_min < model_config.contract_z_min || model_config.z_max > model_config.contract_z_max) {
    RCLCPP_WARN(this->get_logger(),
                "preprocessing.detection_area.z_min/z_max [%.3f, %.3f] extends outside the manifest z range "
                "[%.3f, %.3f]. The node will continue, but inference quality may be degraded.",
                static_cast<double>(model_config.z_min), static_cast<double>(model_config.z_max),
                static_cast<double>(model_config.contract_z_min), static_cast<double>(model_config.contract_z_max));
  }

  if (model_config.x_grid_size <= 0) {
    fail("x_grid_size", "must be greater than zero");
  }
  if (model_config.y_grid_size <= 0) {
    fail("y_grid_size", "must be greater than zero");
  }

  if (model_config.predicted_class_names.empty()) {
    fail("predicted_class_names", "must not be empty");
  }
  for (const auto& name : model_config.predicted_class_names) {
    if (name.empty()) {
      fail("predicted_class_names", "class names must not be empty strings");
    }
  }

  if (model_config.nms_iou_threshold < kMinClassScoreThreshold || model_config.nms_iou_threshold > kMaxClassScoreThreshold) {
    fail("postprocessing.nms.iou_threshold", "must be within [0.0, 1.0]");
  }
  if (model_config.nms_score_threshold.empty()) {
    fail("postprocessing.nms.score_threshold", "must not be empty");
  }
  if (model_config.nms_score_threshold.size() != 1 &&
      model_config.nms_score_threshold.size() != model_config.predicted_class_names.size()) {
    fail("postprocessing.nms.score_threshold", "must contain exactly one value or one per predicted class (got " +
                                                   std::to_string(model_config.nms_score_threshold.size()) + ", expected 1 or " +
                                                   std::to_string(model_config.predicted_class_names.size()) + ")");
  }
  for (double threshold : model_config.nms_score_threshold) {
    if (!isProbability(threshold)) {
      fail("postprocessing.nms.score_threshold", "each entry must be within [0.0, 1.0]");
    }
  }
  if (model_config.nms_max_num_objects < 0) {
    fail("postprocessing.nms.max_num_objects", "must be zero or positive");
  }
  if (model_config.point_feature_normalization_type == "value_threshold" && model_config.point_feature_value_threshold <= 0.0f) {
    fail("preprocessing.point_feature.value_threshold", "must be greater than 0");
  }

  if (model_config.max_num_points <= 0) {
    fail("max_num_points", "must be greater than zero");
  }
  if (model_config.stride.empty()) {
    fail("stride", "must not be empty");
  }
  for (auto stride : model_config.stride) {
    if (stride <= 0) {
      fail("stride", "entries must be positive integers");
    }
  }
  if (model_config.first_up_stride <= 0) {
    fail("first_up_stride", "must be greater than zero");
  }

  if (model_config.no_detection_zone_remove_points) {
    if (!(model_config.no_detection_zone_x_min < model_config.no_detection_zone_x_max) ||
        !(model_config.no_detection_zone_y_min < model_config.no_detection_zone_y_max)) {
      fail("preprocessing.no_detection_zone.remove_points", "requires x_min < x_max and y_min < y_max");
    }
  }

  if (model_config.detection_area_remove_points_outside) {
    if (model_config.detection_area_radius <= kMinDetectionAreaRadius) {
      fail("preprocessing.detection_area.radius", "must be greater than 0 when detection area filtering is enabled");
    }
    if (model_config.detection_area_fov_deg <= kMinDetectionAreaFovDeg ||
        model_config.detection_area_fov_deg > kMaxDetectionAreaFovDeg) {
      fail("preprocessing.detection_area.fov_deg", "must be in the range (0, 360]");
    }
  }
}

rcl_interfaces::msg::SetParametersResult PointCloudObjectDetection::parametersCallback(
    const std::vector<rclcpp::Parameter>& parameters) {
  rcl_interfaces::msg::SetParametersResult result;
  result.successful = true;
  result.reason = "success";
  const Params params_before_update = params_;
  const ModelConfig model_config_before_update = model_config_;
  const pcod_common::NmsConfig nms_config_before_update = nms_config_;
  auto rollback_state = [&]() {
    params_ = params_before_update;
    model_config_ = model_config_before_update;
    nms_config_ = nms_config_before_update;
    syncModelRuntimeConfigFromParams();
  };

  const auto name_in = [](const std::string& name, std::initializer_list<const char*> allowed) {
    return std::any_of(allowed.begin(), allowed.end(), [&](const char* s) { return name == s; });
  };

  bool model_change_on_runtime = false;
  bool publishers_changed = false;
  bool repository_changed = false;
  for (const auto& param : parameters) {
    if (requiresModelReinitializationForParameter(param.get_name())) {
      model_change_on_runtime = true;
    }
    if (name_in(param.get_name(),
                {"preprocessing.no_detection_zone.publish_polygon", "preprocessing.detection_area.publish_polygon",
                 "preprocessing.no_detection_zone.publish_points", "output.model_bounds.publish_polygon",
                 "output.grid_maps.publish_density", "output.grid_maps.publish_occupancy", "output.grid_maps.publish_combined",
                 "output.grid_maps.publish_static"})) {
      publishers_changed = true;
    }
  }

  std::vector<rclcpp::Parameter> applied_parameters;

  {
    std::lock_guard<std::mutex> model_lock(model_mutex_);
    try {
      for (const auto& param : parameters) {
        if (param.get_name() == "prediction.model_repository") {
          params_.model_repository = param.as_string();
          applied_parameters.push_back(param);
          repository_changed = (params_.model_repository != params_before_update.model_repository);
          continue;
        }
        if (param.get_name() == "prediction.model_version") {
          params_.model_version = param.as_string();
          applied_parameters.push_back(param);
          continue;
        }
        for (auto& auto_reconfigurable_param : auto_reconfigurable_params_) {
          if (param.get_name() == std::get<0>(auto_reconfigurable_param)) {
            std::get<1>(auto_reconfigurable_param)(param);
            applied_parameters.push_back(param);
            break;
          }
        }
      }

      const double cscu = pm::object_access::CONTINUOUS_STATE_COVARIANCE_UNKNOWN;
      sanitizeVarianceVector(params_.variance, cscu);
      if (repository_changed) {
        // All runtime-configurable fields are read from the current params_ inside
        // refreshResolvedModelConfigLocked, so they are preserved across the model switch.
        refreshResolvedModelConfigLocked();
      } else {
        syncModelRuntimeConfigFromParams();
        syncNmsRuntimeConfigFromParams();
      }
      validateParamsOrThrow();
      if (detection_model_) {
        validateModelConfigOrThrow();
      }
    } catch (const std::exception& e) {
      rollback_state();
      result.successful = false;
      result.reason = std::string("Invalid parameter update: ") + e.what();
      RCLCPP_ERROR(this->get_logger(), "%s", result.reason.c_str());
      return result;
    } catch (...) {
      rollback_state();
      result.successful = false;
      result.reason = "Invalid parameter update: unknown validation error";
      RCLCPP_ERROR(this->get_logger(), "%s", result.reason.c_str());
      return result;
    }
  }

  for (const auto& param : applied_parameters) {
    RCLCPP_INFO(this->get_logger(), "Reconfigured parameter '%s' to: %s", param.get_name().c_str(),
                param.value_to_string().c_str());
  }

  // Reinitialize model if runtime-critical configuration changed
  if (model_change_on_runtime) {
    try {
      model_ready_.store(false, std::memory_order_release);
      initializeModel();
      RCLCPP_INFO(this->get_logger(), "Successfully reinitialized model with name: %s, version: %s", params_.model_name.c_str(),
                  params_.model_version.c_str());
    } catch (const std::exception& e) {
      {
        std::lock_guard<std::mutex> model_lock(model_mutex_);
        rollback_state();
        model_ready_.store(detection_model_ != nullptr, std::memory_order_release);
      }
      result.successful = false;
      result.reason = "Failed to reinitialize model: " + std::string(e.what());
      RCLCPP_ERROR(this->get_logger(), "Failed to reinitialize model: %s", e.what());
      return result;
    }
  }

  // Defer publisher reconfiguration to normal execution context.
  // Some transport publishers may declare/set parameters internally, which is forbidden in set-parameter callbacks.
  if (publishers_changed) {
    publishers_update_pending_.store(true, std::memory_order_release);
  }

  return result;
}

void PointCloudObjectDetection::refreshResolvedModelConfigLocked() {
  const Params params_snapshot = params_;
  std::string model_name;
  pcod_common::NmsConfig new_nms_config;
  auto new_model_config = loadModelConfig(params_snapshot, model_name, new_nms_config);
  validateModelConfigOrThrow(new_model_config);
  params_.model_name = model_name;
  model_config_ = std::move(new_model_config);
  nms_config_ = std::move(new_nms_config);
}

void PointCloudObjectDetection::initializeModel() {
  std::lock_guard<std::mutex> model_lock(model_mutex_);
  if (params_.model_name.empty() || params_.model_version.empty() || model_config_.predicted_class_names.empty()) {
    refreshResolvedModelConfigLocked();
  }

  const Params params_snapshot = params_;
  const std::string model_name = params_.model_name;
  const std::string model_version = params_.model_version;
  ModelConfig new_model_config = model_config_;
  std::unique_ptr<triton_cpp::TritonInterface> new_triton_interface;
  std::unique_ptr<Model> new_detection_model;
  if (params_snapshot.cuda_input_shm && model_config_.preprocessing_backend != "cuda") {
    RCLCPP_WARN(this->get_logger(),
                "prediction.cuda_input_shm is enabled but preprocessing.backend='%s'. "
                "CUDA input shared memory is only used with preprocessing.backend='cuda'; "
                "continuing with the normal CPU input transport.",
                model_config_.preprocessing_backend.c_str());
  }

  // Retry Triton connection until the ROS context is shutting down.
  constexpr auto kRetryDelay = std::chrono::seconds(1);
  while (rclcpp::ok(this->get_node_base_interface()->get_context())) {
    try {
      new_triton_interface = std::make_unique<triton_cpp::TritonInterface>(
          model_name, model_version, params_snapshot.server_url, params_snapshot.use_shm, false, false,
          params_snapshot.triton_client_timeout_s,
          params_snapshot.cuda_input_shm && model_config_.preprocessing_backend == "cuda");
      break;
    } catch (const std::exception& e) {
      RCLCPP_WARN(this->get_logger(), "Failed to connect to Triton server '%s' for model '%s:%s': %s. Retrying in %.1fs.",
                  params_snapshot.server_url.c_str(), model_name.c_str(), model_version.c_str(), e.what(),
                  std::chrono::duration<double>(kRetryDelay).count());
      rclcpp::sleep_for(kRetryDelay);
    }
  }

  if (!new_triton_interface) {
    throw std::runtime_error("Shutdown requested while waiting for Triton connection");
  }

  // log model info
  const std::string model_info = new_triton_interface->getModelInfo();
  std::cout << model_info << std::endl;

  // create model architecture
  PBODModel::validateInterface(*new_triton_interface);
  new_detection_model = std::make_unique<PBODModel>(*new_triton_interface.get(), new_model_config);
  new_triton_interface->initInOutputs(new_detection_model->getSpecialOutputShapes());
  triton_interface_ = std::move(new_triton_interface);
  detection_model_ = std::move(new_detection_model);
  model_ready_.store(true, std::memory_order_release);
}

void PointCloudObjectDetection::setupPublishers() {
  Params params_snapshot;
  {
    std::lock_guard<std::mutex> model_lock(model_mutex_);
    params_snapshot = params_;
  }

  std::lock_guard<std::mutex> publishers_lock(publishers_mutex_);
  // create no-detection zone polygon publisher if enabled
  if (params_snapshot.no_detection_zone_publish_polygon) {
    if (!no_detection_zone_pub_) {
      no_detection_zone_pub_ = create_publisher<geometry_msgs::msg::PolygonStamped>(kNoDetectionZoneTopic, 1);
      RCLCPP_INFO(this->get_logger(), "Publishing no-detection zone polygon on '%s'", no_detection_zone_pub_->get_topic_name());
    }
  } else {
    no_detection_zone_pub_.reset();
  }

  // create detection area polygon publisher if requested
  if (params_snapshot.detection_area_publish_polygon) {
    if (!detection_area_pub_) {
      detection_area_pub_ = create_publisher<geometry_msgs::msg::PolygonStamped>(kDetectionAreaTopic, 1);
      RCLCPP_INFO(this->get_logger(), "Publishing detection area polygon on '%s'", detection_area_pub_->get_topic_name());
    }
  } else {
    detection_area_pub_.reset();
  }

  // create model bounds polygon publisher if requested
  if (params_snapshot.model_bounds_publish_polygon) {
    if (!model_bounds_pub_) {
      model_bounds_pub_ = create_publisher<geometry_msgs::msg::PolygonStamped>(kModelBoundsTopic, 1);
      RCLCPP_INFO(this->get_logger(), "Publishing model bounds polygon on '%s'", model_bounds_pub_->get_topic_name());
    }
  } else {
    model_bounds_pub_.reset();
  }

  if (params_snapshot.publish_density_grid_map) {
    if (!density_grid_pub_) {
      density_grid_pub_ = create_publisher<nav_msgs::msg::OccupancyGrid>(kDensityGridMapTopic, 1);
      RCLCPP_INFO(this->get_logger(), "Publishing density grid map on '%s'", density_grid_pub_->get_topic_name());
    }
  } else {
    density_grid_pub_.reset();
  }

  if (params_snapshot.publish_occupancy_grid_map) {
    if (!occupancy_grid_pub_) {
      occupancy_grid_pub_ = create_publisher<nav_msgs::msg::OccupancyGrid>(kOccupancyGridMapTopic, 1);
      RCLCPP_INFO(this->get_logger(), "Publishing occupancy grid map on '%s'", occupancy_grid_pub_->get_topic_name());
    }
  } else {
    occupancy_grid_pub_.reset();
  }

  if (params_snapshot.publish_combined_grid_map) {
    if (!combined_grid_pub_) {
      combined_grid_pub_ = create_publisher<nav_msgs::msg::OccupancyGrid>(kCombinedGridMapTopic, 1);
      RCLCPP_INFO(this->get_logger(), "Publishing combined grid map on '%s'", combined_grid_pub_->get_topic_name());
    }
  } else {
    combined_grid_pub_.reset();
  }

  if (params_snapshot.publish_static_grid_map) {
    if (!static_grid_pub_) {
      static_grid_pub_ = create_publisher<nav_msgs::msg::OccupancyGrid>(kStaticGridMapTopic, 1);
      RCLCPP_INFO(this->get_logger(), "Publishing static grid map on '%s'", static_grid_pub_->get_topic_name());
    }
  } else {
    static_grid_pub_.reset();
  }

  // create no-detection zone raw points publisher if enabled
  if (params_snapshot.no_detection_zone_publish_points) {
    if (!no_detection_zone_points_publisher_) {
      std::string topic_name = this->get_node_topics_interface()->resolve_topic_name(kNoDetectionZonePointsTopic);
      no_detection_zone_points_publisher_ =
          std::make_shared<point_cloud_transport::Publisher>(point_cloud_transport_->advertise(topic_name, 1));
      RCLCPP_INFO(this->get_logger(), "Publishing no-detection zone points to '%s'",
                  no_detection_zone_points_publisher_->getTopic().c_str());
    }
  } else {
    no_detection_zone_points_publisher_.reset();
  }
}

void PointCloudObjectDetection::setup() {
  // Preload manifest-derived model config before any parameter-driven transport setup.
  // point_cloud_transport may set parameters during subscribe/advertise, and those callbacks
  // validate NMS settings against the predicted class list from the manifest.
  {
    std::lock_guard<std::mutex> model_lock(model_mutex_);
    refreshResolvedModelConfigLocked();
  }

  // create a callback for dynamic parameter configuration
  parameters_callback_ = this->add_on_set_parameters_callback(
      std::bind(&PointCloudObjectDetection::parametersCallback, this, std::placeholders::_1));

  // create a transform buffer and listener
  tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
  tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

  // Use a non-owning node handle to avoid ownership cycles with transport plugins.
  auto node_handle = std::shared_ptr<rclcpp::Node>(this, [](rclcpp::Node*) {});
  point_cloud_transport_ = std::make_unique<point_cloud_transport::PointCloudTransport>(node_handle);

  // create subscriber and publisher
  std::string resolved_input_topic = this->get_node_topics_interface()->resolve_topic_name(kInputTopic);
  subscriber_ = std::make_shared<point_cloud_transport::Subscriber>(point_cloud_transport_->subscribe(
      resolved_input_topic, 1, std::bind(&PointCloudObjectDetection::predict, this, std::placeholders::_1),
      std::shared_ptr<void>()));
  publisher_ = create_publisher<pm::msg::ObjectList>(kOutputTopic, 1);
  RCLCPP_INFO(this->get_logger(), "Subscribed to '%s'", subscriber_->getTopic().c_str());
  RCLCPP_INFO(this->get_logger(), "Publishing to '%s'", publisher_->get_topic_name());

  // Setup all publishers
  setupPublishers();

  // Initialize the model after transport endpoints exist so runtime dependencies
  // are exercised during startup even if Triton is temporarily unavailable.
  initializeModel();
}

void PointCloudObjectDetection::processPointCloud(const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg,
                                                  const ModelConfig& model_config,
                                                  const Params& params,
                                                  PointCloud& point_cloud) {
  const auto prepared = preparePointCloudInput(msg, model_config, params, *tf_buffer_, this->get_logger());
  decodePreparedPointCloudToPcl(prepared, point_cloud);
}

void PointCloudObjectDetection::boxesToObjectList(const std::vector<BoundingBox>& bboxes,
                                                  const ModelConfig& model_config,
                                                  const Params& params,
                                                  pm::msg::ObjectList& object_list) {
  // iterate over all boxes
  for (int idx = 0; idx < int(bboxes.size()); ++idx) {
    pm::msg::Object object;
    pm::object_access::initializeState(object, pm::msg::ISCACTR::MODEL_ID);

    // Set sensor ID
    object.state.sensor_id.push_back(params.sensor_id);

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
    std::vector<double> variance(params.variance.begin(), params.variance.end());
    pm::object_access::setContinuousStateCovarianceDiagonal(object, variance);

    // get class probability
    for (auto& classification_entry : bboxes[idx].classification) {
      pm::msg::ObjectClassification cls;
      cls.probability = classification_entry.score;
      const std::string& cls_name = model_config.predicted_class_names[classification_entry.class_idx];
      cls.type = model_config.class_mapping_.at(cls_name);
      if (auto it = std::find_if(object.state.classifications.begin(), object.state.classifications.end(),
                                 [&cls](const pm::msg::ObjectClassification& c) { return c.type == cls.type; });
          it != object.state.classifications.end()) {
        // Note that PBOD stores logits, so accumulate with logaddexp.
        it->probability = logaddexp(it->probability, cls.probability);
      } else {
        // Else, add this class
        object.state.classifications.push_back(cls);
      }
    }
    // normalize class probabilities and remove unlikely classes
    sanitize_classifications(object.state.classifications, params.output_class_score_threshold, true);
    // add to object list
    object_list.objects.push_back(object);
  }

  // transform if required
  if (!params.output_frame.empty() && params.inference_frame != params.output_frame) {
    geometry_msgs::msg::TransformStamped t;
    try {
      t = tf_buffer_->lookupTransform(params.output_frame, params.inference_frame, object_list.header.stamp);
    } catch (tf2::TransformException& e) {
      RCLCPP_ERROR(this->get_logger(),
                   "Cannot transform Object list: Transformation from inference_frame (%s) to output_frame "
                   "(%s) not found: %s",
                   params.inference_frame.c_str(), params.output_frame.c_str(), e.what());
      return;
    }
    pm::msg::ObjectList object_list_trans;
    tf2::doTransform(object_list, object_list_trans, t);
    object_list = object_list_trans;
    object_list.header.frame_id = params.output_frame;
  } else if (!params.output_frame.empty()) {
    // if inference = output frame, just change frame in header, but no transform needed
    object_list.header.frame_id = params.output_frame;
  } else if (!params.inference_frame.empty()) {
    // if output frame is empty and inference frame is not empty, just change frame in header, but no transform needed
    object_list.header.frame_id = params.inference_frame;
  }
}

void PointCloudObjectDetection::predict(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& msg) {
  // Guard the whole callback so unexpected runtime errors drop only this frame instead of crashing the node.
  try {
    if (publishers_update_pending_.exchange(false, std::memory_order_acq_rel)) {
      setupPublishers();
    }

    if (!model_ready_.load(std::memory_order_acquire)) {
      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                           "Detection model is not ready yet. Dropping incoming point cloud frame.");
      return;
    }

    Params params_snapshot;
    ModelConfig model_config_snapshot;
    pcod_common::NmsConfig nms_config_snapshot;
    {
      std::lock_guard<std::mutex> model_lock(model_mutex_);
      params_snapshot = params_;
      model_config_snapshot = model_config_;
      nms_config_snapshot = nms_config_;
    }

    rclcpp::Publisher<geometry_msgs::msg::PolygonStamped>::SharedPtr no_detection_zone_pub;
    rclcpp::Publisher<geometry_msgs::msg::PolygonStamped>::SharedPtr detection_area_pub;
    rclcpp::Publisher<geometry_msgs::msg::PolygonStamped>::SharedPtr model_bounds_pub;
    rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr density_grid_pub;
    rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr occupancy_grid_pub;
    rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr combined_grid_pub;
    rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr static_grid_pub;
    std::shared_ptr<point_cloud_transport::Publisher> no_detection_zone_points_publisher;
    {
      std::lock_guard<std::mutex> publishers_lock(publishers_mutex_);
      no_detection_zone_pub = no_detection_zone_pub_;
      detection_area_pub = detection_area_pub_;
      model_bounds_pub = model_bounds_pub_;
      density_grid_pub = density_grid_pub_;
      occupancy_grid_pub = occupancy_grid_pub_;
      combined_grid_pub = combined_grid_pub_;
      static_grid_pub = static_grid_pub_;
      no_detection_zone_points_publisher = no_detection_zone_points_publisher_;
    }

    std::vector<std::chrono::time_point<std::chrono::high_resolution_clock>> timestamps = {
        std::chrono::high_resolution_clock::now()};  // index: 0, start timer

    auto header = msg->header;
    const PreparedPointCloudInput prepared_input =
        preparePointCloudInput(msg, model_config_snapshot, params_snapshot, *tf_buffer_, this->get_logger());
    const bool need_point_cloud = params_snapshot.no_detection_zone_publish_points && params_snapshot.no_detection_zone_enabled &&
                                  no_detection_zone_points_publisher;
    PointCloud point_cloud;
    if (need_point_cloud) {
      decodePreparedPointCloudToPcl(prepared_input, point_cloud);
    }
    timestamps.push_back(std::chrono::high_resolution_clock::now());  // index: 1, after pcl preprocessing

    // Optionally publish raw points inside the no-detection zone
    if (need_point_cloud) {
      PointCloud ndz_points;
      ndz_points.header.frame_id =
          params_snapshot.inference_frame.empty() ? msg->header.frame_id : params_snapshot.inference_frame;
      pcl_conversions::toPCL(msg->header.stamp, ndz_points.header.stamp);
      const double x_min = params_snapshot.no_detection_zone_x_min;
      const double x_max = params_snapshot.no_detection_zone_x_max;
      const double y_min = params_snapshot.no_detection_zone_y_min;
      const double y_max = params_snapshot.no_detection_zone_y_max;
      for (const auto& p : point_cloud) {
        if (p.x >= x_min && p.x <= x_max && p.y >= y_min && p.y <= y_max) {
          ndz_points.push_back(p);
        }
      }
      sensor_msgs::msg::PointCloud2 ros_msg;
      pcl::toROSMsg(ndz_points, ros_msg);
      ros_msg.header.stamp = msg->header.stamp;
      ros_msg.header.frame_id = ndz_points.header.frame_id;
      no_detection_zone_points_publisher->publish(ros_msg);
    }

    // Publish the no-detection zone polygon if requested and enabled
    if (params_snapshot.no_detection_zone_publish_polygon && params_snapshot.no_detection_zone_enabled && no_detection_zone_pub) {
      geometry_msgs::msg::PolygonStamped polygon_message;
      polygon_message.header.stamp = msg->header.stamp;
      // Use inference_frame if set; otherwise use point cloud frame
      polygon_message.header.frame_id =
          params_snapshot.inference_frame.empty() ? msg->header.frame_id : params_snapshot.inference_frame;

      geometry_msgs::msg::Point32 polygon_point;
      // (x_min, y_min)
      polygon_point.x = static_cast<float>(params_snapshot.no_detection_zone_x_min);
      polygon_point.y = static_cast<float>(params_snapshot.no_detection_zone_y_min);
      polygon_point.z = 0.0f;
      polygon_message.polygon.points.push_back(polygon_point);
      // (x_max, y_min)
      polygon_point.x = static_cast<float>(params_snapshot.no_detection_zone_x_max);
      polygon_point.y = static_cast<float>(params_snapshot.no_detection_zone_y_min);
      polygon_message.polygon.points.push_back(polygon_point);
      // (x_max, y_max)
      polygon_point.x = static_cast<float>(params_snapshot.no_detection_zone_x_max);
      polygon_point.y = static_cast<float>(params_snapshot.no_detection_zone_y_max);
      polygon_message.polygon.points.push_back(polygon_point);
      // (x_min, y_max)
      polygon_point.x = static_cast<float>(params_snapshot.no_detection_zone_x_min);
      polygon_point.y = static_cast<float>(params_snapshot.no_detection_zone_y_max);
      polygon_message.polygon.points.push_back(polygon_point);

      no_detection_zone_pub->publish(polygon_message);
    }

    // Publish the model bounds rectangle as polygon if requested
    if (params_snapshot.model_bounds_publish_polygon && model_bounds_pub) {
      geometry_msgs::msg::PolygonStamped polygon_message;
      polygon_message.header.stamp = msg->header.stamp;
      polygon_message.header.frame_id =
          params_snapshot.inference_frame.empty() ? msg->header.frame_id : params_snapshot.inference_frame;

      geometry_msgs::msg::Point32 p;
      // (x_min, y_min)
      p.x = static_cast<float>(model_config_snapshot.x_min);
      p.y = static_cast<float>(model_config_snapshot.y_min);
      p.z = 0.0f;
      polygon_message.polygon.points.push_back(p);
      // (x_max, y_min)
      p.x = static_cast<float>(model_config_snapshot.x_max);
      p.y = static_cast<float>(model_config_snapshot.y_min);
      polygon_message.polygon.points.push_back(p);
      // (x_max, y_max)
      p.x = static_cast<float>(model_config_snapshot.x_max);
      p.y = static_cast<float>(model_config_snapshot.y_max);
      polygon_message.polygon.points.push_back(p);
      // (x_min, y_max)
      p.x = static_cast<float>(model_config_snapshot.x_min);
      p.y = static_cast<float>(model_config_snapshot.y_max);
      polygon_message.polygon.points.push_back(p);

      model_bounds_pub->publish(polygon_message);
    }

    // Publish the detection area sector polygon if requested and enabled
    if (params_snapshot.detection_area_publish_polygon && params_snapshot.detection_area_enabled && detection_area_pub) {
      geometry_msgs::msg::PolygonStamped polygon_message;
      polygon_message.header.stamp = msg->header.stamp;
      polygon_message.header.frame_id =
          params_snapshot.inference_frame.empty() ? msg->header.frame_id : params_snapshot.inference_frame;

      const double sector_center_x = params_snapshot.detection_area_center_x;
      const double sector_center_y = params_snapshot.detection_area_center_y;
      const double sector_radius = params_snapshot.detection_area_radius;
      const double sector_bearing_deg = params_snapshot.detection_area_bearing_deg;
      const double sector_fov_deg = params_snapshot.detection_area_fov_deg;
      const int num_segments =
          std::max(static_cast<int>(kMinDetectionAreaNumSegments), params_snapshot.detection_area_num_segments);

      // Build a sector polygon first; for a full 360-degree FOV, use only the perimeter so no spoke closes to center.
      std::vector<geometry_msgs::msg::Point32> sector_poly;
      {
        const double start_angle_deg = sector_bearing_deg - 0.5 * sector_fov_deg;
        const double end_angle_deg = sector_bearing_deg + 0.5 * sector_fov_deg;
        constexpr double kFullCircleToleranceDeg = 1e-3;
        const bool is_full_circle = std::abs(sector_fov_deg - kMaxDetectionAreaFovDeg) <= kFullCircleToleranceDeg;
        if (!is_full_circle) {
          geometry_msgs::msg::Point32 pt;
          pt.x = static_cast<float>(sector_center_x);
          pt.y = static_cast<float>(sector_center_y);
          pt.z = 0.0f;
          sector_poly.push_back(pt);
        }

        const int arc_samples = is_full_circle ? num_segments : num_segments + 1;
        for (int i = 0; i < arc_samples; ++i) {
          const double angle_deg = start_angle_deg + (end_angle_deg - start_angle_deg) * static_cast<double>(i) / num_segments;
          const double angle_rad = angle_deg * M_PI / 180.0;
          geometry_msgs::msg::Point32 arc_point;
          arc_point.x = static_cast<float>(sector_center_x + sector_radius * std::cos(angle_rad));
          arc_point.y = static_cast<float>(sector_center_y + sector_radius * std::sin(angle_rad));
          arc_point.z = 0.0f;
          sector_poly.push_back(arc_point);
        }
      }

      // Sutherland-Hodgman clipping: each pass clips against one rectangle side and carries
      // generated intersection vertices forward, yielding a robust clipped polygon.
      auto clip_poly_to_halfplane = [](const std::vector<geometry_msgs::msg::Point32>& poly, auto inside, auto intersect) {
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

      const double rx_min = static_cast<double>(model_config_snapshot.x_min);
      const double rx_max = static_cast<double>(model_config_snapshot.x_max);
      const double ry_min = static_cast<double>(model_config_snapshot.y_min);
      const double ry_max = static_cast<double>(model_config_snapshot.y_max);

      // Explicit line intersections keep clipping numerically stable for nearly axis-aligned edges.
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

      // Clip in order: left, right, bottom, top. Order is arbitrary for convex clip polygons,
      // but keeping it fixed makes behavior easier to reason about.
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
        detection_area_pub->publish(polygon_message);
      }
    }

    // pointcloudToBoxes
    // Timing layout is fixed to keep debug logs comparable across runs and with model-internal timings.
    // 0=start, 1=after point cloud conversion, 2=after model input prep, 3=after inference,
    // 4=after output decode, 5=after NMS, 6=after geometric filtering, 7=after msg conversion, 8=after publish.
    // Index 2 and 3 are appended by the model implementation itself.
    std::vector<BoundingBox> center_boxes;
    std::optional<AuxiliaryGridMap> density_grid_map;
    std::optional<AuxiliaryGridMap> occupancy_grid_map;
    std::optional<AuxiliaryGridMap> combined_grid_map;
    std::optional<AuxiliaryGridMap> static_grid_map;
    const bool need_density_grid_map = params_snapshot.publish_density_grid_map || params_snapshot.publish_combined_grid_map ||
                                       params_snapshot.publish_static_grid_map;
    const bool need_occupancy_grid_map = params_snapshot.publish_occupancy_grid_map ||
                                         params_snapshot.publish_combined_grid_map || params_snapshot.publish_static_grid_map;
    std::size_t used_points = 0;
    try {
      std::lock_guard<std::mutex> model_lock(model_mutex_);
      if (!detection_model_) {
        throw std::runtime_error("Detection model is not initialized");
      }
      detection_model_->setAuxiliaryGridMapRequest(AuxiliaryGridMapRequest{need_density_grid_map, need_occupancy_grid_map});
      auto* pbod_model = dynamic_cast<PBODModel*>(detection_model_.get());
      const bool can_use_direct_preprocess = pbod_model != nullptr && !need_point_cloud;
      if (can_use_direct_preprocess) {
        pbod_model->prepareModelInputFromPointCloud2(*prepared_input.msg, prepared_input.x_offset, prepared_input.y_offset,
                                                     prepared_input.z_offset, prepared_input.feature_offset,
                                                     prepared_input.feature_datatype, prepared_input.needs_swap, false);
        timestamps.push_back(std::chrono::high_resolution_clock::now());
        center_boxes = detection_model_->inferAndDecode(timestamps);
      } else {
        if (!need_point_cloud) {
          decodePreparedPointCloudToPcl(prepared_input, point_cloud);
        }
        center_boxes = (*detection_model_)(point_cloud, timestamps);
      }
      used_points = detection_model_->getFilteredInputPointCount();
      if (need_density_grid_map) {
        density_grid_map = detection_model_->getDensityGridMap();
      }
      if (need_occupancy_grid_map) {
        occupancy_grid_map = detection_model_->getOccupancyGridMap();
      }
    } catch (const std::exception& e) {
      RCLCPP_WARN(this->get_logger(), "Lost Triton connection for '%s:%s' on server '%s': %s. Attempting to reconnect.",
                  params_snapshot.model_name.c_str(), params_snapshot.model_version.c_str(), params_snapshot.server_url.c_str(),
                  e.what());
      try {
        initializeModel();
        RCLCPP_WARN(this->get_logger(), "Recovered Triton connection for '%s:%s'. Dropping current frame and continuing.",
                    params_snapshot.model_name.c_str(), params_snapshot.model_version.c_str());
      } catch (const std::exception& reconnect_error) {
        RCLCPP_ERROR(this->get_logger(), "Triton reconnection attempt failed: %s", reconnect_error.what());
      }
      return;
    }
    timestamps.push_back(std::chrono::high_resolution_clock::now());  // index: 4, after output tensor creation

    // Build derived maps from raw model outputs before applying per-map display gains.
    // The combination formulas require probability-scale inputs [0, 1]; applying gain
    // beforehand would distort those semantics.
    const std::string grid_map_source_frame = resolveAuxiliaryGridMapSourceFrame(params_snapshot, header);
    const std::string grid_map_target_frame = resolveAuxiliaryGridMapTargetFrame(params_snapshot, grid_map_source_frame);
    std::optional<geometry_msgs::msg::Pose> grid_map_origin;
    if (density_grid_map.has_value() && occupancy_grid_map.has_value()) {
      if (params_snapshot.publish_combined_grid_map) {
        combined_grid_map = buildCombinedAuxiliaryGridMap(*density_grid_map, *occupancy_grid_map);
      }
      if (params_snapshot.publish_static_grid_map) {
        static_grid_map = buildStaticAuxiliaryGridMap(*density_grid_map, *occupancy_grid_map);
      }
    }
    if (params_snapshot.publish_density_grid_map && !density_grid_map.has_value()) {
      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                           "Density grid map publication was requested, but the model did not provide "
                           "density_logits output.");
    }
    if (params_snapshot.publish_occupancy_grid_map && !occupancy_grid_map.has_value()) {
      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                           "Occupancy grid map publication was requested, but the model did not provide "
                           "occupancy_logits output.");
    }
    if (params_snapshot.publish_combined_grid_map && !combined_grid_map.has_value()) {
      if (!density_grid_map.has_value() || !occupancy_grid_map.has_value()) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                             "Combined grid map publication was requested, but density_logits and occupancy_logits "
                             "outputs are not both available.");
      } else {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                             "Failed to build combined grid map from incompatible model outputs.");
      }
    }
    if (params_snapshot.publish_static_grid_map && !static_grid_map.has_value()) {
      if (!density_grid_map.has_value() || !occupancy_grid_map.has_value()) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                             "Static grid map publication was requested, but density_logits and occupancy_logits "
                             "outputs are not both available.");
      } else {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                             "Failed to build static grid map from incompatible model outputs.");
      }
    }
    if (params_snapshot.publish_density_grid_map && density_grid_map.has_value()) {
      applyAuxiliaryGridMapMasks(*density_grid_map, params_snapshot);
      applyAuxiliaryGridMapGain(*density_grid_map, params_snapshot.density_grid_map_gain);
    }
    if (params_snapshot.publish_occupancy_grid_map && occupancy_grid_map.has_value()) {
      applyAuxiliaryGridMapMasks(*occupancy_grid_map, params_snapshot);
      applyAuxiliaryGridMapGain(*occupancy_grid_map, params_snapshot.occupancy_grid_map_gain);
    }
    if (params_snapshot.publish_combined_grid_map && combined_grid_map.has_value()) {
      applyAuxiliaryGridMapMasks(*combined_grid_map, params_snapshot);
      applyAuxiliaryGridMapGain(*combined_grid_map, params_snapshot.combined_grid_map_gain);
    }
    if (params_snapshot.publish_static_grid_map && static_grid_map.has_value()) {
      applyAuxiliaryGridMapMasks(*static_grid_map, params_snapshot);
      applyAuxiliaryGridMapGain(*static_grid_map, params_snapshot.static_grid_map_gain);
    }
    if ((params_snapshot.publish_density_grid_map && density_grid_map.has_value()) ||
        (params_snapshot.publish_occupancy_grid_map && occupancy_grid_map.has_value()) ||
        (params_snapshot.publish_combined_grid_map && combined_grid_map.has_value()) ||
        (params_snapshot.publish_static_grid_map && static_grid_map.has_value())) {
      const AuxiliaryGridMap* reference_grid_map = nullptr;
      if (density_grid_map.has_value()) {
        reference_grid_map = &*density_grid_map;
      } else if (occupancy_grid_map.has_value()) {
        reference_grid_map = &*occupancy_grid_map;
      } else if (combined_grid_map.has_value()) {
        reference_grid_map = &*combined_grid_map;
      } else if (static_grid_map.has_value()) {
        reference_grid_map = &*static_grid_map;
      }

      if (reference_grid_map != nullptr) {
        grid_map_origin =
            transformAuxiliaryGridMapOriginPose(makeAuxiliaryGridMapOriginPose(*reference_grid_map), header,
                                                grid_map_source_frame, grid_map_target_frame, *tf_buffer_, this->get_logger());
      }
    }
    if (params_snapshot.publish_density_grid_map && density_grid_map.has_value() && density_grid_pub &&
        grid_map_origin.has_value()) {
      if (auto density_message =
              buildAuxiliaryOccupancyGridMessage(*density_grid_map, header, grid_map_target_frame, *grid_map_origin);
          density_message.has_value()) {
        density_grid_pub->publish(std::move(*density_message));
      } else {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                             "Failed to build density grid map message from model output.");
      }
    }
    if (params_snapshot.publish_occupancy_grid_map && occupancy_grid_map.has_value() && occupancy_grid_pub &&
        grid_map_origin.has_value()) {
      if (auto occupancy_message =
              buildAuxiliaryOccupancyGridMessage(*occupancy_grid_map, header, grid_map_target_frame, *grid_map_origin);
          occupancy_message.has_value()) {
        occupancy_grid_pub->publish(std::move(*occupancy_message));
      } else {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                             "Failed to build occupancy grid map message from model output.");
      }
    }
    if (params_snapshot.publish_combined_grid_map && combined_grid_map.has_value() && combined_grid_pub &&
        grid_map_origin.has_value()) {
      if (auto combined_message =
              buildAuxiliaryOccupancyGridMessage(*combined_grid_map, header, grid_map_target_frame, *grid_map_origin);
          combined_message.has_value()) {
        combined_grid_pub->publish(std::move(*combined_message));
      } else {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                             "Failed to build combined grid map message from model output.");
      }
    }
    if (params_snapshot.publish_static_grid_map && static_grid_map.has_value() && static_grid_pub &&
        grid_map_origin.has_value()) {
      if (auto static_message =
              buildAuxiliaryOccupancyGridMessage(*static_grid_map, header, grid_map_target_frame, *grid_map_origin);
          static_message.has_value()) {
        static_grid_pub->publish(std::move(*static_message));
      } else {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                             "Failed to build static grid map message from model output.");
      }
    }

    const std::size_t boxes_before_nms = center_boxes.size();

    // non-maximum suppression
    pcod_common::ApplyRotatedNms(center_boxes, nms_config_snapshot);
    timestamps.push_back(std::chrono::high_resolution_clock::now());  // index: 5, after nms
    const std::size_t boxes_after_nms = center_boxes.size();

    // filter detections intersecting no-detection zone (inference_frame)
    if (params_snapshot.no_detection_zone_enabled) {
      BoundingBox no_detection_rectangle;
      const float rect_x_min = static_cast<float>(params_snapshot.no_detection_zone_x_min);
      const float rect_x_max = static_cast<float>(params_snapshot.no_detection_zone_x_max);
      const float rect_y_min = static_cast<float>(params_snapshot.no_detection_zone_y_min);
      const float rect_y_max = static_cast<float>(params_snapshot.no_detection_zone_y_max);
      no_detection_rectangle.center = {0.5f * (rect_x_min + rect_x_max), 0.5f * (rect_y_min + rect_y_max)};
      no_detection_rectangle.z = 0.0f;
      no_detection_rectangle.length = (rect_x_max - rect_x_min);
      no_detection_rectangle.width = (rect_y_max - rect_y_min);
      no_detection_rectangle.height = 0.0f;  // not used for 2D intersection
      no_detection_rectangle.yaw = 0.0f;     // axis-aligned rectangle in inference_frame
      no_detection_rectangle.existence_probability = 1.0f;

      // Remove any overlapping detection, not just detections whose center lies in the zone.
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
    if (params_snapshot.detection_area_enabled && params_snapshot.detection_area_filter_detections) {
      const double sector_center_x = params_snapshot.detection_area_center_x;
      const double sector_center_y = params_snapshot.detection_area_center_y;
      const double sector_radius = params_snapshot.detection_area_radius;
      const double sector_bearing_rad = params_snapshot.detection_area_bearing_deg * M_PI / 180.0;
      const double sector_fov_rad = params_snapshot.detection_area_fov_deg * M_PI / 180.0;
      const double z_min = params_snapshot.detection_area_z_min;
      const double z_max = params_snapshot.detection_area_z_max;
      const bool require_complete_box_inside = (params_snapshot.detection_area_filter_mode == "complete");

      auto is_point_inside_sector = [&](double x, double y) -> bool {
        const double delta_x = x - sector_center_x;
        const double delta_y = y - sector_center_y;
        const double squared_distance_to_center = delta_x * delta_x + delta_y * delta_y;
        if (squared_distance_to_center > sector_radius * sector_radius) return false;
        double angle_to_center = std::atan2(delta_y, delta_x);
        double angle_offset = angle_to_center - sector_bearing_rad;
        // Normalize to [-pi, pi] so wrap-around at +/-pi does not misclassify boundary angles.
        while (angle_offset > M_PI) angle_offset -= 2.0 * M_PI;
        while (angle_offset < -M_PI) angle_offset += 2.0 * M_PI;
        return std::abs(angle_offset) <= (sector_fov_rad * 0.5 + 1e-9);
      };

      auto is_z_center_inside_range = [&](const BoundingBox& bbox) -> bool {
        return static_cast<double>(bbox.z) >= z_min && static_cast<double>(bbox.z) <= z_max;
      };

      auto is_z_extent_inside_range = [&](const BoundingBox& bbox) -> bool {
        const double half_height = 0.5 * static_cast<double>(bbox.height);
        const double box_z_min = static_cast<double>(bbox.z) - half_height;
        const double box_z_max = static_cast<double>(bbox.z) + half_height;
        return box_z_min >= z_min && box_z_max <= z_max;
      };

      auto is_bounding_box_inside_sector = [&](const BoundingBox& bbox) -> bool {
        if (!require_complete_box_inside) {
          // "center": cheaper and less strict; keeps boxes with centroids inside the XY sector and z range.
          return is_point_inside_sector(bbox.center[0], bbox.center[1]) && is_z_center_inside_range(bbox);
        }
        // "complete": conservative; all 4 oriented box corners and the full vertical extent must remain inside.
        auto vertices = bbox.rectangle_vertices();
        for (const auto& vertex : vertices) {
          if (!is_point_inside_sector(vertex.x, vertex.y)) return false;
        }
        return is_z_extent_inside_range(bbox);
      };

      auto num_boxes_before = center_boxes.size();
      center_boxes.erase(std::remove_if(center_boxes.begin(), center_boxes.end(),
                                        [&](const BoundingBox& bbox) { return !is_bounding_box_inside_sector(bbox); }),
                         center_boxes.end());
      auto removed = num_boxes_before - center_boxes.size();
      if (removed > 0) {
        RCLCPP_INFO(this->get_logger(), "Filtered %zu detections outside detection area (%s)", removed,
                    params_snapshot.detection_area_filter_mode.c_str());
      }
    }
    timestamps.push_back(std::chrono::high_resolution_clock::now());  // index: 6, after detection filters

    // boxesToObjectList
    pm::msg::ObjectList::UniquePtr object_list = std::make_unique<pm::msg::ObjectList>();
    object_list->header = header;
    boxesToObjectList(center_boxes, model_config_snapshot, params_snapshot, *object_list);
    timestamps.push_back(std::chrono::high_resolution_clock::now());  // index: 7, after object-list conversion

    // get size for logging, as publishing invalidates the message here
    std::size_t size = object_list->objects.size();

    // publish output message
    publisher_->publish(std::move(object_list));
    timestamps.push_back(std::chrono::high_resolution_clock::now());  // index: 8, after publish

    // log processing
    std::chrono::duration<double> inference_time = timestamps[3] - timestamps[2];
    std::chrono::duration<double> total_time = timestamps.back() - timestamps.front();
    RCLCPP_INFO(this->get_logger(), "%ld objects detected in %.3fs (inference %.3fs)", size, total_time.count(),
                inference_time.count());
    const auto to_ms = [](const auto& duration) { return std::chrono::duration<double, std::milli>(duration).count(); };
    const char* event_name = "pcod_timing";
    RCLCPP_DEBUG(this->get_logger(),
                 "%s points=%zu used_points=%zu boxes_in=%zu boxes_nms=%zu boxes_out=%zu objects=%zu "
                 "e2e_ms=%.3f pcl_pre_ms=%.3f model_input_ms=%.3f infer_ms=%.3f decode_ms=%.3f "
                 "nms_ms=%.3f filter_ms=%.3f boxes_to_msg_ms=%.3f publish_ms=%.3f",
                 event_name, prepared_input.total_points, used_points, boxes_before_nms, boxes_after_nms, center_boxes.size(),
                 size, to_ms(timestamps[8] - timestamps[0]), to_ms(timestamps[1] - timestamps[0]),
                 to_ms(timestamps[2] - timestamps[1]), to_ms(timestamps[3] - timestamps[2]), to_ms(timestamps[4] - timestamps[3]),
                 to_ms(timestamps[5] - timestamps[4]), to_ms(timestamps[6] - timestamps[5]), to_ms(timestamps[7] - timestamps[6]),
                 to_ms(timestamps[8] - timestamps[7]));
  } catch (const std::exception& e) {
    RCLCPP_ERROR(this->get_logger(), "Failed to process point cloud frame: %s", e.what());
  } catch (...) {
    RCLCPP_ERROR(this->get_logger(), "Failed to process point cloud frame: unknown exception");
  }
}

// Transition callback for state configuring
// Lifecycle callbacks removed in regular node
}  // namespace point_cloud_object_detection

RCLCPP_COMPONENTS_REGISTER_NODE(point_cloud_object_detection::PointCloudObjectDetection)
