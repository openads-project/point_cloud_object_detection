#include "point_cloud_object_detection/PBODModel.hpp"

namespace point_cloud_object_detection {
PBODModel::PBODModel(triton_cpp::TritonInterface& triton_interface, ModelConfig& model_config)
    : Model(triton_interface),
      model_config_{model_config},
      input_name_point_features_{SAVED_MODEL_INPUT_NAME_POINT_FEATURES},
      input_name_pillar_ids_{SAVED_MODEL_INPUT_NAME_PILLAR_IDS},
      input_name_valid_mask_{SAVED_MODEL_INPUT_NAME_VALID_MASK},
      input_name_pillar_masks_{SAVED_MODEL_INPUT_NAME_PILLAR_MASKS},
      input_name_pillar_indices_{SAVED_MODEL_INPUT_NAME_PILLAR_INDICES},
      output_name_focal_{SAVED_MODEL_OUTPUT_NAME_FOCAL},
      output_name_reg_{SAVED_MODEL_OUTPUT_NAME_REG},
      output_name_class_{SAVED_MODEL_OUTPUT_NAME_CLASS},
      output_name_size_{SAVED_MODEL_OUTPUT_NAME_SIZE},
      x_min_{model_config_.pillar_map_range[0][0]},
      x_max_{model_config_.pillar_map_range[0][1]},
      y_min_{model_config_.pillar_map_range[1][0]},
      y_max_{model_config_.pillar_map_range[1][1]},
      z_min_{model_config_.pillar_map_range[2][0]},
      z_max_{model_config_.pillar_map_range[2][1]},
      voxel_x_{static_cast<float>(model_config_.voxel_x)},
      voxel_y_{static_cast<float>(model_config_.voxel_y)},
      voxel_z_{static_cast<float>(model_config_.voxel_z)},
      normalization_type_{pcod_common::ParsePointFeatureNormalizationType(
          model_config_.point_feature_normalization_type)},
      intensity_threshold_{model_config_.point_feature_intensity_threshold},
      min_intensity_{model_config_.point_feature_min_intensity},
      max_intensity_{model_config_.point_feature_max_intensity},
      norm_epsilon_{model_config_.point_feature_norm_epsilon},
      zero_intensity_{model_config_.zero_intensity},
      max_num_points_{model_config_.max_num_points},
      num_point_features_{model_config_.num_point_features},
      preprocessed_feature_dim_{num_point_features_ + 17},
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
  pillar_indices_.resize(static_cast<std::size_t>(num_pillars_) * 2);
  const int grid_x = static_cast<int>(model_config_.pillar_map_size[0]);
  const int grid_y = static_cast<int>(model_config_.pillar_map_size[1]);
  for (int ix = 0; ix < grid_x; ++ix) {
    for (int iy = 0; iy < grid_y; ++iy) {
      const std::size_t idx = static_cast<std::size_t>(ix * grid_y + iy) * 2;
      pillar_indices_[idx] = ix;
      pillar_indices_[idx + 1] = iy;
    }
  }
  pillar_grid_ = pcod_common::BuildPillarGrid(
      {grid_x, grid_y},
      {{{model_config_.pillar_map_range[0][0], model_config_.pillar_map_range[0][1]},
        {model_config_.pillar_map_range[1][0], model_config_.pillar_map_range[1][1]},
        {model_config_.pillar_map_range[2][0], model_config_.pillar_map_range[2][1]}}},
      model_config_.first_up_stride,
      stride);
}

void PBODModel::setAdditionalPointFeatures(const float* feature_values, std::size_t point_count,
                                           std::size_t feature_stride) {
  external_point_features_ = feature_values;
  external_point_feature_count_ = point_count;
  external_point_feature_stride_ = feature_stride;
}

std::map<std::string, std::vector<int64_t>> PBODModel::getSpecialOutputShapes() {
  const int64_t num_pillars = static_cast<int64_t>(pillar_grid_.grid_x * pillar_grid_.grid_y);
  const int64_t num_classes = static_cast<int64_t>(model_config_.predicted_class_names.size());
  return {{output_name_reg_, {num_pillars, 7 * num_classes}}, {output_name_size_, {num_pillars, 3 * num_classes}}};
}

void PBODModel::setupModelInput(const PointCloud& point_cloud) {
  auto point_features_map =
      triton_interface_.getInputTensor<float>(input_name_point_features_, max_num_points_, preprocessed_feature_dim_);
  auto pillar_ids_map = triton_interface_.getInputTensor<int64_t>(input_name_pillar_ids_, max_num_points_);
  auto valid_mask_map = triton_interface_.getInputTensor<bool>(input_name_valid_mask_, max_num_points_);
  auto pillar_masks_map = triton_interface_.getInputTensor<bool>(input_name_pillar_masks_, num_pillars_);
  auto pillar_indices_map = triton_interface_.getInputTensor<int64_t>(input_name_pillar_indices_, num_pillars_, 2);

  point_features_map.setZero();
  valid_mask_map.setZero();
  pillar_masks_map.setZero();
  pillar_indices_map.setZero();

  const int64_t sentinel = static_cast<int64_t>(num_pillars_);
  for (int i = 0; i < max_num_points_; ++i) {
    pillar_ids_map(i) = sentinel;
  }

  for (int i = 0; i < num_pillars_; ++i) {
    const std::size_t idx = static_cast<std::size_t>(i) * 2;
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

  struct SampledPoint {
    Point point;
    std::size_t index;
  };
  std::vector<SampledPoint> sampled;
  sampled.reserve(static_cast<std::size_t>(max_num_points_));

  static thread_local std::mt19937 gen(std::random_device{}());
  int valid_point_count = 0;
  for (int i = 0; i < n_cloud_points; ++i) {
    const auto& point = point_cloud[i];
    if (!point_preprocessor_.IsPointValid(point.x, point.y, point.z)) {
      continue;
    }
    ++valid_point_count;
    if (valid_point_count <= max_num_points_) {
      sampled.push_back({point, static_cast<std::size_t>(i)});
    } else {
      std::uniform_int_distribution<int> dist(1, valid_point_count);
      const int random_idx = dist(gen);
      if (random_idx <= max_num_points_) {
        sampled[static_cast<std::size_t>(random_idx - 1)] = {point, static_cast<std::size_t>(i)};
      }
    }
  }

  const int num_selected = static_cast<int>(sampled.size());
  filtered_input_points_.reserve(num_selected);
  for (const auto& entry : sampled) {
    filtered_input_points_.push_back(entry.point);
  }

  if (normalization_type_ == pcod_common::PointFeatureNormalizationType::kZScore) {
    double sum = 0.0;
    double sum_sq = 0.0;
    for (const auto& entry : sampled) {
      const double value = static_cast<double>(entry.point.intensity);
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
  std::vector<float> pillar_sum(static_cast<std::size_t>(num_pillars_) * 3, 0.0f);
  std::vector<float> pillar_sq_sum(static_cast<std::size_t>(num_pillars_) * 3, 0.0f);
  std::vector<int> pillar_ids_raw(static_cast<std::size_t>(num_selected), -1);

  for (int i = 0; i < num_selected; ++i) {
    const auto& point = sampled[static_cast<std::size_t>(i)].point;
    if (point.x < x_min_ || point.x >= x_max_ || point.y < y_min_ || point.y >= y_max_ || point.z < z_min_ ||
        point.z >= z_max_) {
      continue;
    }

    const int grid_x = static_cast<int>(model_config_.pillar_map_size[0]);
    const int grid_y = static_cast<int>(model_config_.pillar_map_size[1]);
    const int ix = std::max(0, std::min(static_cast<int>((point.x - x_min_) / voxel_x_), grid_x - 1));
    const int iy = std::max(0, std::min(static_cast<int>((point.y - y_min_) / voxel_y_), grid_y - 1));
    const int pillar_id = ix * grid_y + iy;
    pillar_ids_raw[static_cast<std::size_t>(i)] = pillar_id;

    const std::size_t count_offset = static_cast<std::size_t>(pillar_id);
    ++pillar_counts[count_offset];
    const std::size_t sum_offset = count_offset * 3;
    pillar_sum[sum_offset + 0] += point.x;
    pillar_sum[sum_offset + 1] += point.y;
    pillar_sum[sum_offset + 2] += point.z;
    pillar_sq_sum[sum_offset + 0] += point.x * point.x;
    pillar_sq_sum[sum_offset + 1] += point.y * point.y;
    pillar_sq_sum[sum_offset + 2] += point.z * point.z;
  }

  for (int i = 0; i < num_selected; ++i) {
    const auto& entry = sampled[static_cast<std::size_t>(i)];
    const int pillar_id = pillar_ids_raw[static_cast<std::size_t>(i)];
    if (pillar_id < 0) {
      continue;
    }

    const std::size_t count_offset = static_cast<std::size_t>(pillar_id);
    const int32_t count = pillar_counts[count_offset];
    if (count <= 0) {
      continue;
    }

    const std::size_t sum_offset = count_offset * 3;
    const float inv_count = 1.0f / static_cast<float>(count);
    const float mean_x = pillar_sum[sum_offset + 0] * inv_count;
    const float mean_y = pillar_sum[sum_offset + 1] * inv_count;
    const float mean_z = pillar_sum[sum_offset + 2] * inv_count;
    const float mean_x2 = pillar_sq_sum[sum_offset + 0] * inv_count;
    const float mean_y2 = pillar_sq_sum[sum_offset + 1] * inv_count;
    const float mean_z2 = pillar_sq_sum[sum_offset + 2] * inv_count;

    const auto& point = entry.point;
    const int grid_x = static_cast<int>(model_config_.pillar_map_size[0]);
    const int grid_y = static_cast<int>(model_config_.pillar_map_size[1]);
    const int ix = std::max(0, std::min(static_cast<int>((point.x - x_min_) / voxel_x_), grid_x - 1));
    const int iy = std::max(0, std::min(static_cast<int>((point.y - y_min_) / voxel_y_), grid_y - 1));

    const float center_x = x_min_ + (static_cast<float>(ix) + 0.5f) * voxel_x_;
    const float center_y = y_min_ + (static_cast<float>(iy) + 0.5f) * voxel_y_;
    const float center_z = z_min_ + (z_max_ - z_min_) * 0.5f;

    const float f_cluster_x = point.x - mean_x;
    const float f_cluster_y = point.y - mean_y;
    const float f_cluster_z = point.z - mean_z;
    const float f_center_x = point.x - center_x;
    const float f_center_y = point.y - center_y;
    const float f_center_z = point.z - center_z;
    const float r = std::sqrt(point.x * point.x + point.y * point.y);
    const float z_rel = point.z - z_min_;
    const float eps = 1e-6f;
    const float inv_r = 1.0f / (r > eps ? r : eps);
    const float theta = std::atan2(point.y, point.x);
    const float sin_theta = std::sin(theta);
    const float cos_theta = std::cos(theta);
    const float var_x = std::max(mean_x2 - mean_x * mean_x, 0.0f);
    const float var_y = std::max(mean_y2 - mean_y * mean_y, 0.0f);
    const float var_z = std::max(mean_z2 - mean_z * mean_z, 0.0f);

    float raw_feature0 = point_preprocessor_.NormalizeIntensity(point.intensity);
    float raw_feature1 = 0.0f;
    if (num_point_features_ > 1) {
      const float* extra = getExtraFeatures(entry.index);
      if (extra != nullptr) {
        raw_feature1 = extra[0];
      }
    }

    int offset = 0;
    point_features_map(i, offset++) = point.x;
    point_features_map(i, offset++) = point.y;
    point_features_map(i, offset++) = point.z;
    point_features_map(i, offset++) = r;
    point_features_map(i, offset++) = z_rel;
    point_features_map(i, offset++) = inv_r;
    point_features_map(i, offset++) = sin_theta;
    point_features_map(i, offset++) = cos_theta;
    point_features_map(i, offset++) = var_x;
    point_features_map(i, offset++) = var_y;
    point_features_map(i, offset++) = var_z;
    point_features_map(i, offset++) = raw_feature0;
    if (num_point_features_ > 1) {
      point_features_map(i, offset++) = raw_feature1;
    }
    point_features_map(i, offset++) = f_cluster_x;
    point_features_map(i, offset++) = f_cluster_y;
    point_features_map(i, offset++) = f_cluster_z;
    point_features_map(i, offset++) = f_center_x;
    point_features_map(i, offset++) = f_center_y;
    point_features_map(i, offset++) = f_center_z;

    valid_mask_map(i) = true;
    pillar_ids_map(i) = static_cast<int64_t>(pillar_id);
  }

  for (int i = 0; i < num_pillars_; ++i) {
    if (pillar_counts[static_cast<std::size_t>(i)] > 0) {
      pillar_masks_map(i) = true;
    }
  }
}

std::vector<BoundingBox> PBODModel::modelOutputToBoxes() {
  const int num_pillars = pillar_grid_.grid_x * pillar_grid_.grid_y;
  const int num_classes = static_cast<int>(model_config_.predicted_class_names.size());
  auto class_logits = triton_interface_.getOutputTensor<float>(output_name_class_, num_pillars, num_classes);
  auto size_posterior = triton_interface_.getOutputTensor<float>(output_name_size_, num_pillars, 3 * num_classes);
  auto focal_logits = triton_interface_.getOutputTensor<float>(output_name_focal_, num_pillars);
  auto reg_logits = triton_interface_.getOutputTensor<float>(output_name_reg_, num_pillars, 7 * num_classes);

  pcod_common::PbodOutputsView view;
  view.focal_logits = focal_logits.data();
  view.size_posterior = size_posterior.data();
  view.class_logits = class_logits.data();
  view.reg_logits = reg_logits.data();
  view.num_pillars = num_pillars;
  view.num_classes = num_classes;
  view.reg_dim = 7;

  pcod_common::PbodPostprocessConfig config;
  config.class_names = model_config_.predicted_class_names;
  config.score_thresholds.clear();
  for (double value : model_config_.nms_score_threshold) {
    config.score_thresholds.push_back(static_cast<float>(value));
  }

  return pcod_common::DecodePbod(view, pillar_grid_, config);
}

}  // namespace point_cloud_object_detection
