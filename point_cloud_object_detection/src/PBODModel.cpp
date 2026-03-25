#include "point_cloud_object_detection/PBODModel.hpp"

namespace point_cloud_object_detection {

void PBODModel::validateInterface(const triton_cpp::TritonInterface& triton_interface) {
  if (triton_interface.nInputs() != kExpectedInputNames.size()) {
    throw std::runtime_error("PBOD model interface mismatch: expected " + std::to_string(kExpectedInputNames.size()) +
                             " inputs but Triton reports " + std::to_string(triton_interface.nInputs()));
  }
  if (triton_interface.nOutputs() != kExpectedOutputNames.size()) {
    throw std::runtime_error("PBOD model interface mismatch: expected " + std::to_string(kExpectedOutputNames.size()) +
                             " outputs but Triton reports " + std::to_string(triton_interface.nOutputs()));
  }

  for (const char* input_name : kExpectedInputNames) {
    try {
      (void)triton_interface.getInputShape(input_name);
    } catch (const std::invalid_argument& e) {
      throw std::runtime_error("PBOD model is missing expected input tensor '" + std::string(input_name) +
                               "': " + e.what());
    }
  }

  for (const char* output_name : kExpectedOutputNames) {
    try {
      (void)triton_interface.getOutputShape(output_name);
    } catch (const std::invalid_argument& e) {
      throw std::runtime_error("PBOD model is missing expected output tensor '" + std::string(output_name) +
                               "': " + e.what());
    }
  }
}

PBODModel::PBODModel(triton_cpp::TritonInterface& triton_interface, const ModelConfig& model_config)
    : Model(triton_interface),
      model_config_{model_config},
      x_min_{model_config_.pillar_map_range[0][0]},
      x_max_{model_config_.pillar_map_range[0][1]},
      y_min_{model_config_.pillar_map_range[1][0]},
      y_max_{model_config_.pillar_map_range[1][1]},
      z_min_{model_config_.pillar_map_range[2][0]},
      z_max_{model_config_.pillar_map_range[2][1]},
      voxel_x_{static_cast<float>(model_config_.voxel_x)},
      voxel_y_{static_cast<float>(model_config_.voxel_y)},
      voxel_z_{static_cast<float>(model_config_.voxel_z)},
      normalization_type_{
          pcod_common::ParsePointFeatureNormalizationType(model_config_.point_feature_normalization_type)},
      intensity_threshold_{model_config_.point_feature_intensity_threshold},
      min_intensity_{model_config_.point_feature_min_intensity},
      max_intensity_{model_config_.point_feature_max_intensity},
      norm_epsilon_{model_config_.point_feature_norm_epsilon},
      zero_intensity_{model_config_.zero_intensity},
      max_num_points_{model_config_.max_num_points},
      preprocessed_feature_dim_{kPreprocessedFeatureDim},
      num_pillars_{static_cast<int>(model_config_.pillar_map_size[0] * model_config_.pillar_map_size[1])},
      remove_points_in_zone_{model_config_.no_detection_zone_remove_points},
      nd_x_min_{static_cast<float>(model_config_.no_detection_zone_x_min)},
      nd_x_max_{static_cast<float>(model_config_.no_detection_zone_x_max)},
      nd_y_min_{static_cast<float>(model_config_.no_detection_zone_y_min)},
      nd_y_max_{static_cast<float>(model_config_.no_detection_zone_y_max)},
      det_area_remove_outside_{model_config_.detection_area_remove_points_outside},
      da_cx_{static_cast<float>(model_config_.detection_area_center_x)},
      da_cy_{static_cast<float>(model_config_.detection_area_center_y)},
      da_radius_{static_cast<float>(model_config_.detection_area_radius)},
      da_bearing_rad_{static_cast<float>(model_config_.detection_area_bearing_deg * M_PI / 180.0)},
      da_fov_rad_{static_cast<float>(model_config_.detection_area_fov_deg * M_PI / 180.0)},
      point_preprocessor_({x_min_,
                           x_max_,
                           y_min_,
                           y_max_,
                           z_min_,
                           z_max_,
                           normalization_type_,
                           intensity_threshold_,
                           min_intensity_,
                           max_intensity_,
                           norm_epsilon_,
                           zero_intensity_,
                           remove_points_in_zone_,
                           nd_x_min_,
                           nd_x_max_,
                           nd_y_min_,
                           nd_y_max_,
                           det_area_remove_outside_,
                           da_cx_,
                           da_cy_,
                           da_radius_,
                           da_bearing_rad_,
                           da_fov_rad_}) {
  const int stride = model_config_.stride.empty() ? 1 : static_cast<int>(model_config_.stride[0]);
  pillar_indices_.resize(static_cast<std::size_t>(num_pillars_) * kPillarIndexDim);
  const int grid_x = static_cast<int>(model_config_.pillar_map_size[0]);
  const int grid_y = static_cast<int>(model_config_.pillar_map_size[1]);
  for (int ix = 0; ix < grid_x; ++ix) {
    for (int iy = 0; iy < grid_y; ++iy) {
      const std::size_t idx = static_cast<std::size_t>(ix * grid_y + iy) * kPillarIndexDim;
      pillar_indices_[idx] = ix;
      pillar_indices_[idx + 1] = iy;
    }
  }
  pillar_grid_ =
      pcod_common::BuildPillarGrid({grid_x, grid_y},
                                   {{{model_config_.pillar_map_range[0][0], model_config_.pillar_map_range[0][1]},
                                     {model_config_.pillar_map_range[1][0], model_config_.pillar_map_range[1][1]},
                                     {model_config_.pillar_map_range[2][0], model_config_.pillar_map_range[2][1]}}},
                                   model_config_.first_up_stride, stride);
}

std::map<std::string, std::vector<int64_t>> PBODModel::getSpecialOutputShapes() {
  const int64_t num_pillars = static_cast<int64_t>(pillar_grid_.grid_x * pillar_grid_.grid_y);
  const int64_t num_classes = static_cast<int64_t>(model_config_.predicted_class_names.size());
  return {{kOutputNameReg, {num_pillars, kRegressionValuesPerClass * num_classes}},
          {kOutputNameSize, {num_pillars, kSizeValuesPerClass * num_classes}}};
}

void PBODModel::setupModelInput(const PointCloud& point_cloud) {
  const int grid_y = static_cast<int>(model_config_.pillar_map_size[1]);
  const float inv_voxel_x = 1.0f / voxel_x_;
  const float inv_voxel_y = 1.0f / voxel_y_;

  auto point_features_map =
      triton_interface_.getInputTensor<float>(kInputNamePointFeatures, max_num_points_, preprocessed_feature_dim_);
  auto pillar_ids_map = triton_interface_.getInputTensor<int64_t>(kInputNamePillarIds, max_num_points_);
  auto valid_mask_map = triton_interface_.getInputTensor<bool>(kInputNameValidMask, max_num_points_);
  auto pillar_masks_map = triton_interface_.getInputTensor<bool>(kInputNamePillarMasks, num_pillars_);
  auto pillar_indices_map =
      triton_interface_.getInputTensor<int64_t>(kInputNamePillarIndices, num_pillars_, kPillarIndexDim);

  point_features_map.setZero();
  valid_mask_map.setZero();
  pillar_masks_map.setZero();

  const int64_t sentinel = static_cast<int64_t>(num_pillars_);
  pillar_ids_map.setConstant(sentinel);

  for (int i = 0; i < num_pillars_; ++i) {
    const std::size_t idx = static_cast<std::size_t>(i) * kPillarIndexDim;
    pillar_indices_map(i, 0) = pillar_indices_[idx];
    pillar_indices_map(i, 1) = pillar_indices_[idx + 1];
  }

  filtered_input_points_.clear();
  filtered_input_points_.reserve(std::min(static_cast<int>(point_cloud.size()), max_num_points_));

  const int n_cloud_points = static_cast<int>(point_cloud.size());
  if (n_cloud_points == 0) {
    RCLCPP_DEBUG(rclcpp::get_logger("PBODModel"), "Input point cloud is empty.");
    return;
  }

  static thread_local std::mt19937 gen(std::random_device{}());
  static thread_local std::uniform_int_distribution<int> reservoir_dist;
  int valid_point_count = 0;
  for (int i = 0; i < n_cloud_points; ++i) {
    const auto& point = point_cloud[i];
    if (!point_preprocessor_.IsPointValid(point.x, point.y, point.z)) {
      continue;
    }
    ++valid_point_count;
    if (valid_point_count <= max_num_points_) {
      filtered_input_points_.push_back(point);
    } else {
      reservoir_dist.param(std::uniform_int_distribution<int>::param_type(1, valid_point_count));
      const int random_idx = reservoir_dist(gen);
      if (random_idx <= max_num_points_) {
        filtered_input_points_[static_cast<std::size_t>(random_idx - 1)] = point;
      }
    }
  }

  const int num_selected = static_cast<int>(filtered_input_points_.size());

  if (normalization_type_ == pcod_common::PointFeatureNormalizationType::kZScore) {
    double sum = 0.0;
    double sum_sq = 0.0;
    for (const auto& point : filtered_input_points_) {
      const double value = static_cast<double>(point.intensity);
      sum += value;
      sum_sq += value * value;
    }
    const double count = static_cast<double>(std::max(1, num_selected));
    const double mean = sum / count;
    const double variance = std::max(0.0, sum_sq / count - mean * mean);
    const double stddev = std::sqrt(variance);
    point_preprocessor_.SetZScoreStats(static_cast<float>(mean), static_cast<float>(stddev));
  }

  std::vector<int32_t> pillar_counts(static_cast<std::size_t>(num_pillars_), 0);
  std::vector<float> pillar_sum(static_cast<std::size_t>(num_pillars_) * kPointCoordinateDim, 0.0f);
  std::vector<float> pillar_sq_sum(static_cast<std::size_t>(num_pillars_) * kPointCoordinateDim, 0.0f);
  std::vector<int> pillar_ids_raw(static_cast<std::size_t>(num_selected), -1);
  std::vector<int> pillar_ix_raw(static_cast<std::size_t>(num_selected), -1);
  std::vector<int> pillar_iy_raw(static_cast<std::size_t>(num_selected), -1);

  for (int i = 0; i < num_selected; ++i) {
    const auto& point = filtered_input_points_[static_cast<std::size_t>(i)];
    // Points come from IsPointValid(), so x/y/z are already in model bounds.
    const int ix = static_cast<int>((point.x - x_min_) * inv_voxel_x);
    const int iy = static_cast<int>((point.y - y_min_) * inv_voxel_y);
    const int pillar_id = ix * grid_y + iy;
    pillar_ids_raw[static_cast<std::size_t>(i)] = pillar_id;
    pillar_ix_raw[static_cast<std::size_t>(i)] = ix;
    pillar_iy_raw[static_cast<std::size_t>(i)] = iy;

    const std::size_t count_offset = static_cast<std::size_t>(pillar_id);
    ++pillar_counts[count_offset];
    if (pillar_counts[count_offset] == 1) {
      pillar_masks_map(pillar_id) = true;
    }
    const std::size_t sum_offset = count_offset * kPointCoordinateDim;
    pillar_sum[sum_offset + 0] += point.x;
    pillar_sum[sum_offset + 1] += point.y;
    pillar_sum[sum_offset + 2] += point.z;
    pillar_sq_sum[sum_offset + 0] += point.x * point.x;
    pillar_sq_sum[sum_offset + 1] += point.y * point.y;
    pillar_sq_sum[sum_offset + 2] += point.z * point.z;
  }

  std::vector<float> pillar_mean(static_cast<std::size_t>(num_pillars_) * kPointCoordinateDim, 0.0f);
  std::vector<float> pillar_var(static_cast<std::size_t>(num_pillars_) * kPointCoordinateDim, 0.0f);
  for (int i = 0; i < num_pillars_; ++i) {
    const std::size_t count_offset = static_cast<std::size_t>(i);
    const int32_t count = pillar_counts[count_offset];
    if (count <= 0) {
      continue;
    }
    const std::size_t sum_offset = count_offset * kPointCoordinateDim;
    const float inv_count = 1.0f / static_cast<float>(count);
    const float mean_x = pillar_sum[sum_offset + 0] * inv_count;
    const float mean_y = pillar_sum[sum_offset + 1] * inv_count;
    const float mean_z = pillar_sum[sum_offset + 2] * inv_count;
    const float mean_x2 = pillar_sq_sum[sum_offset + 0] * inv_count;
    const float mean_y2 = pillar_sq_sum[sum_offset + 1] * inv_count;
    const float mean_z2 = pillar_sq_sum[sum_offset + 2] * inv_count;
    pillar_mean[sum_offset + 0] = mean_x;
    pillar_mean[sum_offset + 1] = mean_y;
    pillar_mean[sum_offset + 2] = mean_z;
    pillar_var[sum_offset + 0] = std::max(mean_x2 - mean_x * mean_x, 0.0f);
    pillar_var[sum_offset + 1] = std::max(mean_y2 - mean_y * mean_y, 0.0f);
    pillar_var[sum_offset + 2] = std::max(mean_z2 - mean_z * mean_z, 0.0f);
  }

  constexpr float kRadiusEpsilon = 1e-6f;
  const float center_z = z_min_ + (z_max_ - z_min_) * 0.5f;
  for (int i = 0; i < num_selected; ++i) {
    const int pillar_id = pillar_ids_raw[static_cast<std::size_t>(i)];
    if (pillar_id < 0) {
      continue;
    }

    const std::size_t sum_offset = static_cast<std::size_t>(pillar_id) * kPointCoordinateDim;
    const auto& point = filtered_input_points_[static_cast<std::size_t>(i)];
    const int ix = pillar_ix_raw[static_cast<std::size_t>(i)];
    const int iy = pillar_iy_raw[static_cast<std::size_t>(i)];

    const float center_x = x_min_ + (static_cast<float>(ix) + 0.5f) * voxel_x_;
    const float center_y = y_min_ + (static_cast<float>(iy) + 0.5f) * voxel_y_;
    const float mean_x = pillar_mean[sum_offset + 0];
    const float mean_y = pillar_mean[sum_offset + 1];
    const float mean_z = pillar_mean[sum_offset + 2];

    const float f_cluster_x = point.x - mean_x;
    const float f_cluster_y = point.y - mean_y;
    const float f_cluster_z = point.z - mean_z;
    const float f_center_x = point.x - center_x;
    const float f_center_y = point.y - center_y;
    const float f_center_z = point.z - center_z;
    const float r2 = point.x * point.x + point.y * point.y;
    const float r = std::sqrt(r2);
    const float z_rel = point.z - z_min_;
    const float inv_r = 1.0f / (r > kRadiusEpsilon ? r : kRadiusEpsilon);
    float sin_theta = 0.0f;
    float cos_theta = 1.0f;
    if (r > 0.0f) {
      const float inv_norm = 1.0f / r;
      sin_theta = point.y * inv_norm;
      cos_theta = point.x * inv_norm;
    }
    const float var_x = pillar_var[sum_offset + 0];
    const float var_y = pillar_var[sum_offset + 1];
    const float var_z = pillar_var[sum_offset + 2];

    const float normalized_intensity = point_preprocessor_.NormalizeIntensity(point.intensity);

    point_features_map(i, kFeatureX) = point.x;
    point_features_map(i, kFeatureY) = point.y;
    point_features_map(i, kFeatureZ) = point.z;
    point_features_map(i, kFeatureRadius) = r;
    point_features_map(i, kFeatureZRelative) = z_rel;
    point_features_map(i, kFeatureInverseRadius) = inv_r;
    point_features_map(i, kFeatureSinTheta) = sin_theta;
    point_features_map(i, kFeatureCosTheta) = cos_theta;
    point_features_map(i, kFeatureVarianceX) = var_x;
    point_features_map(i, kFeatureVarianceY) = var_y;
    point_features_map(i, kFeatureVarianceZ) = var_z;
    point_features_map(i, kFeatureIntensity) = normalized_intensity;
    point_features_map(i, kFeatureClusterOffsetX) = f_cluster_x;
    point_features_map(i, kFeatureClusterOffsetY) = f_cluster_y;
    point_features_map(i, kFeatureClusterOffsetZ) = f_cluster_z;
    point_features_map(i, kFeatureCenterOffsetX) = f_center_x;
    point_features_map(i, kFeatureCenterOffsetY) = f_center_y;
    point_features_map(i, kFeatureCenterOffsetZ) = f_center_z;

    valid_mask_map(i) = true;
    pillar_ids_map(i) = static_cast<int64_t>(pillar_id);
  }
}

std::vector<BoundingBox> PBODModel::modelOutputToBoxes() {
  const int num_pillars = pillar_grid_.grid_x * pillar_grid_.grid_y;
  const int num_classes = static_cast<int>(model_config_.predicted_class_names.size());
  auto class_logits = triton_interface_.getOutputTensor<float>(kOutputNameClass, num_pillars, num_classes);
  auto size_posterior =
      triton_interface_.getOutputTensor<float>(kOutputNameSize, num_pillars, kSizeValuesPerClass * num_classes);
  auto focal_logits = triton_interface_.getOutputTensor<float>(kOutputNameFocal, num_pillars);
  auto reg_logits =
      triton_interface_.getOutputTensor<float>(kOutputNameReg, num_pillars, kRegressionValuesPerClass * num_classes);

  pcod_common::PbodOutputsView view;
  view.focal_logits = focal_logits.data();
  view.size_posterior = size_posterior.data();
  view.class_logits = class_logits.data();
  view.reg_logits = reg_logits.data();
  view.num_pillars = num_pillars;
  view.num_classes = num_classes;
  view.reg_dim = kRegressionValuesPerClass;

  pcod_common::PbodPostprocessConfig config;
  config.class_names = model_config_.predicted_class_names;
  config.score_thresholds.clear();
  for (double value : model_config_.nms_score_threshold) {
    config.score_thresholds.push_back(static_cast<float>(value));
  }

  return pcod_common::DecodePbod(view, pillar_grid_, config);
}

}  // namespace point_cloud_object_detection
