// Copyright Institute for Automotive Engineering (ika), RWTH Aachen University
// SPDX-License-Identifier: Apache-2.0

#include "point_cloud_object_detection/PBODModel.hpp"

#include <Eigen/Dense>

#include <sensor_msgs/msg/point_field.hpp>

#include "point_cloud_object_detection/PointCloud2Utils.hpp"
#include "point_cloud_object_detection/PointUtils.hpp"

namespace point_cloud_object_detection {

using PointRecord = pcod_common::PillarPreprocessPoint;

namespace {

constexpr float kDensityTransformScale = 10000.0F;

using FeatureTensorView = Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;
using BoolTensorView = Eigen::Map<Eigen::Matrix<bool, Eigen::Dynamic, 1>>;
using ConstBoolTensorView = Eigen::Map<const Eigen::Matrix<bool, Eigen::Dynamic, 1>>;
using Int64TensorView = Eigen::Map<Eigen::Matrix<std::int64_t, Eigen::Dynamic, 1>>;

bool useCudaPreprocessing(const ModelConfig& model_config) { return model_config.preprocessing_backend == "cuda"; }

float sigmoid(float value) {
  if (value >= 0.0F) {
    const float z = std::exp(-value);
    return 1.0F / (1.0F + z);
  }
  const float z = std::exp(value);
  return z / (1.0F + z);
}

float densityTransform(float value) {
  const float normalized = std::asinh(kDensityTransformScale * value) / std::asinh(kDensityTransformScale);
  return std::clamp(normalized, 0.0F, 1.0F);
}

}  // namespace

void PBODModel::validateInterface(const triton_cpp::TritonInterface& triton_interface) {
  if (triton_interface.nInputs() != kExpectedInputNames.size()) {
    throw std::runtime_error("PBOD model interface mismatch: expected " + std::to_string(kExpectedInputNames.size()) +
                             " inputs but Triton reports " + std::to_string(triton_interface.nInputs()));
  }

  for (const char* input_name : kExpectedInputNames) {
    try {
      (void)triton_interface.getInputShape(input_name);
    } catch (const std::invalid_argument& e) {
      throw std::runtime_error("PBOD model is missing expected input tensor '" + std::string(input_name) + "': " + e.what());
    }
  }

  for (const char* output_name : {kOutputNameFocal, kOutputNameReg, kOutputNameClass, kOutputNameSize}) {
    try {
      (void)triton_interface.getOutputShape(output_name);
    } catch (const std::invalid_argument& e) {
      throw std::runtime_error("PBOD model is missing expected output tensor '" + std::string(output_name) + "': " + e.what());
    }
  }

  bool has_density = true;
  bool has_dynamic_occupancy = true;
  try {
    (void)triton_interface.getOutputShape(kOutputNameDensity);
  } catch (const std::invalid_argument&) {
    has_density = false;
  }
  try {
    (void)triton_interface.getOutputShape(kOutputNameDynamicOccupancy);
  } catch (const std::invalid_argument&) {
    has_dynamic_occupancy = false;
  }
  if (has_density != has_dynamic_occupancy) {
    throw std::runtime_error(
        "PBOD model must expose both density_logits and "
        "dynamic_occupancy_logits together.");
  }
}

PBODModel::PBODModel(triton_cpp::TritonInterface& triton_interface, const ModelConfig& model_config)
    : Model(triton_interface),
      model_config_{model_config},
      postprocess_config_{},
      x_min_{model_config_.pillar_map_range[0][0]},
      x_max_{model_config_.pillar_map_range[0][1]},
      y_min_{model_config_.pillar_map_range[1][0]},
      y_max_{model_config_.pillar_map_range[1][1]},
      z_min_{model_config_.pillar_map_range[2][0]},
      z_max_{model_config_.pillar_map_range[2][1]},
      voxel_x_{static_cast<float>(model_config_.voxel_x)},
      voxel_y_{static_cast<float>(model_config_.voxel_y)},
      normalization_type_{pcod_common::ParsePointFeatureNormalizationType(model_config_.point_feature_normalization_type)},
      value_threshold_{model_config_.point_feature_value_threshold},
      min_value_{model_config_.point_feature_min_value},
      max_value_{model_config_.point_feature_max_value},
      norm_epsilon_{model_config_.point_feature_norm_epsilon},
      max_num_points_{model_config_.max_num_points},
      preprocessed_feature_dim_{kPreprocessedFeatureDim},
      num_pillars_{static_cast<int>(model_config_.pillar_map_size[0] * model_config_.pillar_map_size[1])},
      pillar_id_sentinel_{static_cast<std::int64_t>(num_pillars_)},
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
                           value_threshold_,
                           min_value_,
                           max_value_,
                           norm_epsilon_,
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
  if (useCudaPreprocessing(model_config_) && !cuda_preprocess_context_.isAvailable()) {
    throw std::runtime_error(
        "CUDA preprocessing was requested via preprocessing.backend='cuda', "
        "but CUDA preprocessing support is not "
        "available in this build/runtime");
  }

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
  pillar_grid_ = pcod_common::BuildPillarGrid({grid_x, grid_y},
                                              {{{model_config_.pillar_map_range[0][0], model_config_.pillar_map_range[0][1]},
                                                {model_config_.pillar_map_range[1][0], model_config_.pillar_map_range[1][1]},
                                                {model_config_.pillar_map_range[2][0], model_config_.pillar_map_range[2][1]}}},
                                              model_config_.first_up_stride, stride);
  postprocess_config_.class_names = model_config_.predicted_class_names;
  postprocess_config_.score_thresholds.reserve(model_config_.nms_score_threshold.size());
  for (double value : model_config_.nms_score_threshold) {
    postprocess_config_.score_thresholds.push_back(static_cast<float>(value));
  }
  pillar_counts_.resize(static_cast<std::size_t>(num_pillars_), 0);
  pillar_sum_.resize(static_cast<std::size_t>(num_pillars_) * kPointCoordinateDim, 0.0F);
  pillar_sq_sum_.resize(static_cast<std::size_t>(num_pillars_) * kPointCoordinateDim, 0.0F);
  pillar_mean_.resize(static_cast<std::size_t>(num_pillars_) * kPointCoordinateDim, 0.0F);
  pillar_var_.resize(static_cast<std::size_t>(num_pillars_) * kPointCoordinateDim, 0.0F);
  pillar_ids_raw_.resize(static_cast<std::size_t>(max_num_points_), -1);
  pillar_ix_raw_.resize(static_cast<std::size_t>(max_num_points_), -1);
  pillar_iy_raw_.resize(static_cast<std::size_t>(max_num_points_), -1);
  tensor_reset_state_.active_point_rows.reserve(static_cast<std::size_t>(max_num_points_));
  tensor_reset_state_.active_pillar_ids.reserve(static_cast<std::size_t>(num_pillars_));
  active_pillar_ids_scratch_.reserve(static_cast<std::size_t>(num_pillars_));
  selected_point_records_.reserve(static_cast<std::size_t>(max_num_points_));
  try {
    (void)triton_interface_.getOutputShape(kOutputNameDensity);
    (void)triton_interface_.getOutputShape(kOutputNameDynamicOccupancy);
    has_auxiliary_grid_map_outputs_ = true;
  } catch (const std::invalid_argument&) {
    has_auxiliary_grid_map_outputs_ = false;
  }
}

std::map<std::string, std::vector<int64_t>> PBODModel::getSpecialOutputShapes() {
  const int64_t num_pillars = static_cast<int64_t>(pillar_grid_.grid_x * pillar_grid_.grid_y);
  const int64_t num_classes = static_cast<int64_t>(model_config_.predicted_class_names.size());
  return {{kOutputNameReg, {num_pillars, kRegressionValuesPerClass * num_classes}},
          {kOutputNameSize, {num_pillars, kSizeValuesPerClass * num_classes}}};
}

void PBODModel::setupModelInput(const PointCloud& point_cloud) {
  setupModelInputFromGetter(
      static_cast<int>(point_cloud.size()),
      [&](int i) {
        const auto& point = point_cloud[static_cast<std::size_t>(i)];
        return PointRecord{getPointX(point), getPointY(point), getPointZ(point), getPointIntensity(point)};
      },
      true);
}

void PBODModel::prepareModelInputFromPointCloud2(const sensor_msgs::msg::PointCloud2& point_cloud_msg,
                                                 uint32_t x_offset,
                                                 uint32_t y_offset,
                                                 uint32_t z_offset,
                                                 uint32_t feature_offset,
                                                 uint8_t feature_datatype,
                                                 bool needs_swap,
                                                 bool materialize_filtered_points) {
  setupModelInputFromGetter(
      static_cast<int>(point_cloud_msg.width * point_cloud_msg.height),
      [&](int i) {
        const PointCloud2PointRecord point_record =
            readPointCloud2PointRecord(point_cloud_msg, static_cast<std::size_t>(i), x_offset, y_offset, z_offset, feature_offset,
                                       feature_datatype, needs_swap);
        return PointRecord{point_record.x, point_record.y, point_record.z, point_record.feature};
      },
      materialize_filtered_points);
}

template <typename PointGetter>
void PBODModel::setupModelInputFromGetter(int n_cloud_points, PointGetter get_point, bool materialize_filtered_points) {
  const bool use_cuda_input_shm = useCudaPreprocessing(model_config_) && triton_interface_.usesCudaInputSharedMemory();

  float* point_features_device = nullptr;
  std::int64_t* pillar_ids_device = nullptr;
  bool* valid_mask_device = nullptr;
  bool* pillar_masks_device = nullptr;
  float* point_features_host = nullptr;
  std::int64_t* pillar_ids_host = nullptr;
  bool* valid_mask_host = nullptr;
  bool* pillar_masks_host = nullptr;

  if (use_cuda_input_shm) {
    point_features_device =
        static_cast<float*>(static_cast<void*>(triton_interface_.getInputTensorDevice(kInputNamePointFeatures).first));
    pillar_ids_device =
        static_cast<std::int64_t*>(static_cast<void*>(triton_interface_.getInputTensorDevice(kInputNamePillarIds).first));
    valid_mask_device = static_cast<bool*>(static_cast<void*>(triton_interface_.getInputTensorDevice(kInputNameValidMask).first));
    pillar_masks_device =
        static_cast<bool*>(static_cast<void*>(triton_interface_.getInputTensorDevice(kInputNamePillarMasks).first));
    if (!pillar_indices_initialized_) {
      triton_interface_.copyInputTensorToDevice(kInputNamePillarIndices, pillar_indices_.data(),
                                                pillar_indices_.size() * sizeof(std::int64_t));
      pillar_indices_initialized_ = true;
    }
  } else {
    auto point_features_map =
        triton_interface_.getInputTensor<float>(kInputNamePointFeatures, max_num_points_, preprocessed_feature_dim_);
    auto pillar_ids_map = triton_interface_.getInputTensor<int64_t>(kInputNamePillarIds, max_num_points_);
    auto valid_mask_map = triton_interface_.getInputTensor<bool>(kInputNameValidMask, max_num_points_);
    auto pillar_masks_map = triton_interface_.getInputTensor<bool>(kInputNamePillarMasks, num_pillars_);
    auto pillar_indices_map = triton_interface_.getInputTensor<int64_t>(kInputNamePillarIndices, num_pillars_, kPillarIndexDim);
    point_features_host = point_features_map.data();
    pillar_ids_host = pillar_ids_map.data();
    valid_mask_host = valid_mask_map.data();
    pillar_masks_host = pillar_masks_map.data();

    if (!pillar_indices_initialized_) {
      for (int i = 0; i < num_pillars_; ++i) {
        const std::size_t idx = static_cast<std::size_t>(i) * kPillarIndexDim;
        pillar_indices_map(i, 0) = pillar_indices_[idx];
        pillar_indices_map(i, 1) = pillar_indices_[idx + 1];
      }
      pillar_indices_initialized_ = true;
    }
  }

  selected_point_records_.clear();
  selected_point_records_.reserve(std::min(n_cloud_points, max_num_points_));
  filtered_input_point_count_ = 0;
  if (materialize_filtered_points) {
    filtered_input_points_.clear();
    filtered_input_points_.reserve(std::min(n_cloud_points, max_num_points_));
  } else {
    filtered_input_points_.clear();
  }

  if (n_cloud_points == 0) {
    RCLCPP_DEBUG(rclcpp::get_logger("PBODModel"), "Input point cloud is empty.");
  }

  static thread_local std::mt19937 gen(std::random_device{}());
  static thread_local std::uniform_int_distribution<int> reservoir_dist;
  int valid_point_count = 0;
  for (int i = 0; i < n_cloud_points; ++i) {
    const PointRecord point = get_point(i);
    if (!point_preprocessor_.IsPointValid(point.x, point.y, point.z)) {
      continue;
    }
    ++valid_point_count;
    if (valid_point_count <= max_num_points_) {
      selected_point_records_.push_back(point);
      if (materialize_filtered_points) {
        filtered_input_points_.push_back(makePoint(point.x, point.y, point.z, point.intensity));
      }
    } else {
      reservoir_dist.param(std::uniform_int_distribution<int>::param_type(1, valid_point_count));
      const int random_idx = reservoir_dist(gen);
      if (random_idx <= max_num_points_) {
        const std::size_t replace_idx = static_cast<std::size_t>(random_idx - 1);
        selected_point_records_[replace_idx] = point;
        if (materialize_filtered_points) {
          filtered_input_points_[replace_idx] = makePoint(point.x, point.y, point.z, point.intensity);
        }
      }
    }
  }

  const int num_selected = static_cast<int>(selected_point_records_.size());
  filtered_input_point_count_ = static_cast<std::size_t>(num_selected);

  if (normalization_type_ == pcod_common::PointFeatureNormalizationType::kZScore) {
    double sum = 0.0;
    double sum_sq = 0.0;
    for (const auto& point : selected_point_records_) {
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

  if (!use_cuda_input_shm) {
    const auto resetAllTensors = [&]() {
      auto point_features_map =
          triton_interface_.getInputTensor<float>(kInputNamePointFeatures, max_num_points_, preprocessed_feature_dim_);
      auto pillar_ids_map = triton_interface_.getInputTensor<int64_t>(kInputNamePillarIds, max_num_points_);
      auto valid_mask_map = triton_interface_.getInputTensor<bool>(kInputNameValidMask, max_num_points_);
      auto pillar_masks_map = triton_interface_.getInputTensor<bool>(kInputNamePillarMasks, num_pillars_);
      point_features_map.setZero();
      valid_mask_map.setZero();
      pillar_masks_map.setZero();
      pillar_ids_map.setConstant(pillar_id_sentinel_);
    };

    const auto resetSparseTensors = [&]() {
      FeatureTensorView point_features_view(point_features_host, max_num_points_, preprocessed_feature_dim_);
      BoolTensorView valid_mask_view(valid_mask_host, max_num_points_);
      Int64TensorView pillar_ids_view(pillar_ids_host, max_num_points_);
      BoolTensorView pillar_masks_view(pillar_masks_host, num_pillars_);

      if (!tensor_reset_state_.initialized) {
        resetAllTensors();
        tensor_reset_state_.initialized = true;
        tensor_reset_state_.active_point_rows.clear();
        tensor_reset_state_.active_pillar_ids.clear();
        return;
      }

      for (int row : tensor_reset_state_.active_point_rows) {
        for (int feature_idx = 0; feature_idx < preprocessed_feature_dim_; ++feature_idx) {
          point_features_view(row, feature_idx) = 0.0F;
        }
        valid_mask_view(row) = false;
        pillar_ids_view(row) = pillar_id_sentinel_;
      }
      for (int pillar_id : tensor_reset_state_.active_pillar_ids) {
        pillar_masks_view(pillar_id) = false;
      }
      tensor_reset_state_.active_point_rows.clear();
      tensor_reset_state_.active_pillar_ids.clear();
    };
    resetSparseTensors();
  } else {
    tensor_reset_state_.active_point_rows.clear();
    tensor_reset_state_.active_pillar_ids.clear();
  }

  if (use_cuda_input_shm) {
    if (populateModelInputOnGpuToDevice(point_features_device, pillar_ids_device, valid_mask_device, pillar_masks_device,
                                        num_selected)) {
      return;
    }
    throw std::runtime_error(
        "Failed to populate PBOD input tensors via CUDA shared memory after "
        "prediction.cuda_input_shm was enabled");
  }

  if (useCudaPreprocessing(model_config_) &&
      populateModelInputOnGpu(point_features_host, pillar_ids_host, valid_mask_host, pillar_masks_host, num_selected)) {
    return;
  }

  if (useCudaPreprocessing(model_config_)) {
    throw std::runtime_error(
        "Failed to populate PBOD input tensors via CUDA preprocessing after "
        "preprocessing.backend='cuda' was enabled");
  }

  populateModelInputOnCpu(point_features_host, pillar_ids_host, valid_mask_host, pillar_masks_host, num_selected);
}

void PBODModel::populateModelInputOnCpu(
    float* point_features, std::int64_t* pillar_ids, bool* valid_mask, bool* pillar_masks, int num_selected) {
  FeatureTensorView point_features_view(point_features, max_num_points_, preprocessed_feature_dim_);
  Int64TensorView pillar_ids_view(pillar_ids, max_num_points_);
  BoolTensorView valid_mask_view(valid_mask, max_num_points_);
  BoolTensorView pillar_masks_view(pillar_masks, num_pillars_);

  const float inv_voxel_x = 1.0F / voxel_x_;
  const float inv_voxel_y = 1.0F / voxel_y_;
  const int grid_y = static_cast<int>(model_config_.pillar_map_size[1]);
  const float center_z = z_min_ + (z_max_ - z_min_) * 0.5F;

  for (int pillar_id : active_pillar_ids_scratch_) {
    const std::size_t sum_offset = static_cast<std::size_t>(pillar_id) * kPointCoordinateDim;
    pillar_counts_[static_cast<std::size_t>(pillar_id)] = 0;
    pillar_sum_[sum_offset + 0] = 0.0F;
    pillar_sum_[sum_offset + 1] = 0.0F;
    pillar_sum_[sum_offset + 2] = 0.0F;
    pillar_sq_sum_[sum_offset + 0] = 0.0F;
    pillar_sq_sum_[sum_offset + 1] = 0.0F;
    pillar_sq_sum_[sum_offset + 2] = 0.0F;
    pillar_mean_[sum_offset + 0] = 0.0F;
    pillar_mean_[sum_offset + 1] = 0.0F;
    pillar_mean_[sum_offset + 2] = 0.0F;
    pillar_var_[sum_offset + 0] = 0.0F;
    pillar_var_[sum_offset + 1] = 0.0F;
    pillar_var_[sum_offset + 2] = 0.0F;
  }
  active_pillar_ids_scratch_.clear();

  for (int i = 0; i < num_selected; ++i) {
    const auto& point = selected_point_records_[static_cast<std::size_t>(i)];
    const int ix = static_cast<int>((point.x - x_min_) * inv_voxel_x);
    const int iy = static_cast<int>((point.y - y_min_) * inv_voxel_y);
    const int pillar_id = ix * grid_y + iy;
    pillar_ids_raw_[static_cast<std::size_t>(i)] = pillar_id;
    pillar_ix_raw_[static_cast<std::size_t>(i)] = ix;
    pillar_iy_raw_[static_cast<std::size_t>(i)] = iy;

    const std::size_t count_offset = static_cast<std::size_t>(pillar_id);
    if (pillar_counts_[count_offset] == 0) {
      active_pillar_ids_scratch_.push_back(pillar_id);
      pillar_masks_view(pillar_id) = true;
      tensor_reset_state_.active_pillar_ids.push_back(pillar_id);
    }
    ++pillar_counts_[count_offset];
    const std::size_t sum_offset = count_offset * kPointCoordinateDim;
    pillar_sum_[sum_offset + 0] += point.x;
    pillar_sum_[sum_offset + 1] += point.y;
    pillar_sum_[sum_offset + 2] += point.z;
    pillar_sq_sum_[sum_offset + 0] += point.x * point.x;
    pillar_sq_sum_[sum_offset + 1] += point.y * point.y;
    pillar_sq_sum_[sum_offset + 2] += point.z * point.z;
  }

  for (int pillar_id : active_pillar_ids_scratch_) {
    const std::size_t count_offset = static_cast<std::size_t>(pillar_id);
    const int32_t count = pillar_counts_[count_offset];
    const std::size_t sum_offset = count_offset * kPointCoordinateDim;
    const float inv_count = 1.0F / static_cast<float>(count);
    const float mean_x = pillar_sum_[sum_offset + 0] * inv_count;
    const float mean_y = pillar_sum_[sum_offset + 1] * inv_count;
    const float mean_z = pillar_sum_[sum_offset + 2] * inv_count;
    const float mean_x2 = pillar_sq_sum_[sum_offset + 0] * inv_count;
    const float mean_y2 = pillar_sq_sum_[sum_offset + 1] * inv_count;
    const float mean_z2 = pillar_sq_sum_[sum_offset + 2] * inv_count;
    pillar_mean_[sum_offset + 0] = mean_x;
    pillar_mean_[sum_offset + 1] = mean_y;
    pillar_mean_[sum_offset + 2] = mean_z;
    pillar_var_[sum_offset + 0] = std::max(mean_x2 - mean_x * mean_x, 0.0F);
    pillar_var_[sum_offset + 1] = std::max(mean_y2 - mean_y * mean_y, 0.0F);
    pillar_var_[sum_offset + 2] = std::max(mean_z2 - mean_z * mean_z, 0.0F);
  }

  tensor_reset_state_.active_point_rows.reserve(static_cast<std::size_t>(num_selected));
  for (int i = 0; i < num_selected; ++i) {
    const int pillar_id = pillar_ids_raw_[static_cast<std::size_t>(i)];
    const std::size_t sum_offset = static_cast<std::size_t>(pillar_id) * kPointCoordinateDim;
    const auto& point = selected_point_records_[static_cast<std::size_t>(i)];
    const int ix = pillar_ix_raw_[static_cast<std::size_t>(i)];
    const int iy = pillar_iy_raw_[static_cast<std::size_t>(i)];

    const float center_x = x_min_ + (static_cast<float>(ix) + 0.5F) * voxel_x_;
    const float center_y = y_min_ + (static_cast<float>(iy) + 0.5F) * voxel_y_;
    const float mean_x = pillar_mean_[sum_offset + 0];
    const float mean_y = pillar_mean_[sum_offset + 1];
    const float mean_z = pillar_mean_[sum_offset + 2];

    const float normalized_point_feature = point_preprocessor_.NormalizePointFeature(point.intensity);
    const auto feature_row = makePbodFeatureRow({point.x, point.y, point.z, normalized_point_feature, mean_x, mean_y, mean_z,
                                                 pillar_var_[sum_offset + 0], pillar_var_[sum_offset + 1],
                                                 pillar_var_[sum_offset + 2], center_x, center_y, center_z, z_min_});
    point_features_view(i, kFeatureX) = feature_row[kPbodFeatureX];
    point_features_view(i, kFeatureY) = feature_row[kPbodFeatureY];
    point_features_view(i, kFeatureZ) = feature_row[kPbodFeatureZ];
    point_features_view(i, kFeatureRadius) = feature_row[kPbodFeatureRadius];
    point_features_view(i, kFeatureZRelative) = feature_row[kPbodFeatureZRelative];
    point_features_view(i, kFeatureInverseRadius) = feature_row[kPbodFeatureInverseRadius];
    point_features_view(i, kFeatureSinTheta) = feature_row[kPbodFeatureSinTheta];
    point_features_view(i, kFeatureCosTheta) = feature_row[kPbodFeatureCosTheta];
    point_features_view(i, kFeatureVarianceX) = feature_row[kPbodFeatureVarianceX];
    point_features_view(i, kFeatureVarianceY) = feature_row[kPbodFeatureVarianceY];
    point_features_view(i, kFeatureVarianceZ) = feature_row[kPbodFeatureVarianceZ];
    point_features_view(i, kFeatureIntensity) = feature_row[kPbodFeatureIntensity];
    point_features_view(i, kFeatureClusterOffsetX) = feature_row[kPbodFeatureClusterOffsetX];
    point_features_view(i, kFeatureClusterOffsetY) = feature_row[kPbodFeatureClusterOffsetY];
    point_features_view(i, kFeatureClusterOffsetZ) = feature_row[kPbodFeatureClusterOffsetZ];
    point_features_view(i, kFeatureCenterOffsetX) = feature_row[kPbodFeatureCenterOffsetX];
    point_features_view(i, kFeatureCenterOffsetY) = feature_row[kPbodFeatureCenterOffsetY];
    point_features_view(i, kFeatureCenterOffsetZ) = feature_row[kPbodFeatureCenterOffsetZ];

    valid_mask_view(i) = true;
    pillar_ids_view(i) = static_cast<std::int64_t>(pillar_id);
    tensor_reset_state_.active_point_rows.push_back(i);
  }
}

bool PBODModel::populateModelInputOnGpu(
    float* point_features, std::int64_t* pillar_ids, bool* valid_mask, bool* pillar_masks, int num_selected) {
  pcod_common::PillarPreprocessCudaConfig config;
  config.x_min = x_min_;
  config.x_max = x_max_;
  config.y_min = y_min_;
  config.y_max = y_max_;
  config.z_min = z_min_;
  config.z_max = z_max_;
  config.voxel_x = voxel_x_;
  config.voxel_y = voxel_y_;
  config.value_threshold = value_threshold_;
  config.min_value = min_value_;
  config.max_value = max_value_;
  config.epsilon = norm_epsilon_;
  config.center_z = z_min_ + (z_max_ - z_min_) * 0.5F;
  config.grid_x = static_cast<std::int32_t>(model_config_.pillar_map_size[0]);
  config.grid_y = static_cast<std::int32_t>(model_config_.pillar_map_size[1]);
  config.num_pillars = num_pillars_;
  config.max_num_points = max_num_points_;
  config.feature_dim = preprocessed_feature_dim_;
  config.normalization_type = normalization_type_;
  config.z_score_mean = 0.0F;
  config.z_score_std = 1.0F;

  // Recompute z-score stats directly for the CUDA config without changing
  // point_preprocessor_ semantics.
  if (normalization_type_ == pcod_common::PointFeatureNormalizationType::kZScore) {
    double sum = 0.0;
    double sum_sq = 0.0;
    for (const auto& point : selected_point_records_) {
      const double value = static_cast<double>(point.intensity);
      sum += value;
      sum_sq += value * value;
    }
    const double count = static_cast<double>(std::max(1, num_selected));
    const double mean = sum / count;
    const double variance = std::max(0.0, sum_sq / count - mean * mean);
    config.z_score_mean = static_cast<float>(mean);
    config.z_score_std = std::max(static_cast<float>(std::sqrt(variance)), norm_epsilon_);
  }

  const pcod_common::PillarPreprocessCudaOutputs outputs{point_features, pillar_ids, valid_mask, pillar_masks};
  std::string error_message;
  if (!cuda_preprocess_context_.run(selected_point_records_.data(), num_selected, config, outputs, &error_message)) {
    RCLCPP_ERROR(rclcpp::get_logger("PBODModel"), "CUDA preprocessing failed: %s", error_message.c_str());
    return false;
  }

  tensor_reset_state_.active_point_rows.resize(static_cast<std::size_t>(num_selected));
  for (int i = 0; i < num_selected; ++i) {
    tensor_reset_state_.active_point_rows[static_cast<std::size_t>(i)] = i;
  }
  tensor_reset_state_.active_pillar_ids.clear();
  ConstBoolTensorView pillar_masks_view(pillar_masks, num_pillars_);
  for (int pillar_id = 0; pillar_id < num_pillars_; ++pillar_id) {
    if (pillar_masks_view(pillar_id)) {
      tensor_reset_state_.active_pillar_ids.push_back(pillar_id);
    }
  }
  active_pillar_ids_scratch_.clear();
  return true;
}

bool PBODModel::populateModelInputOnGpuToDevice(
    float* point_features, std::int64_t* pillar_ids, bool* valid_mask, bool* pillar_masks, int num_selected) {
  pcod_common::PillarPreprocessCudaConfig config;
  config.x_min = x_min_;
  config.x_max = x_max_;
  config.y_min = y_min_;
  config.y_max = y_max_;
  config.z_min = z_min_;
  config.z_max = z_max_;
  config.voxel_x = voxel_x_;
  config.voxel_y = voxel_y_;
  config.value_threshold = value_threshold_;
  config.min_value = min_value_;
  config.max_value = max_value_;
  config.epsilon = norm_epsilon_;
  config.center_z = z_min_ + (z_max_ - z_min_) * 0.5F;
  config.grid_x = static_cast<std::int32_t>(model_config_.pillar_map_size[0]);
  config.grid_y = static_cast<std::int32_t>(model_config_.pillar_map_size[1]);
  config.num_pillars = num_pillars_;
  config.max_num_points = max_num_points_;
  config.feature_dim = preprocessed_feature_dim_;
  config.normalization_type = normalization_type_;
  config.z_score_mean = 0.0F;
  config.z_score_std = 1.0F;

  if (normalization_type_ == pcod_common::PointFeatureNormalizationType::kZScore) {
    double sum = 0.0;
    double sum_sq = 0.0;
    for (const auto& point : selected_point_records_) {
      const double value = static_cast<double>(point.intensity);
      sum += value;
      sum_sq += value * value;
    }
    const double count = static_cast<double>(std::max(1, num_selected));
    const double mean = sum / count;
    const double variance = std::max(0.0, sum_sq / count - mean * mean);
    config.z_score_mean = static_cast<float>(mean);
    config.z_score_std = std::max(static_cast<float>(std::sqrt(variance)), norm_epsilon_);
  }

  const pcod_common::PillarPreprocessCudaDeviceOutputs outputs{point_features, pillar_ids, valid_mask, pillar_masks};
  std::string error_message;
  if (!cuda_preprocess_context_.runToDevice(selected_point_records_.data(), num_selected, config, outputs, &error_message)) {
    RCLCPP_ERROR(rclcpp::get_logger("PBODModel"), "CUDA input shared memory preprocessing failed: %s", error_message.c_str());
    return false;
  }

  return true;
}

std::vector<BoundingBox> PBODModel::modelOutputToBoxes() {
  const int num_pillars = pillar_grid_.grid_x * pillar_grid_.grid_y;
  const int num_classes = static_cast<int>(model_config_.predicted_class_names.size());
  auto class_logits = triton_interface_.getOutputTensor<float>(kOutputNameClass, num_pillars, num_classes);
  auto size_posterior = triton_interface_.getOutputTensor<float>(kOutputNameSize, num_pillars, kSizeValuesPerClass * num_classes);
  auto focal_logits = triton_interface_.getOutputTensor<float>(kOutputNameFocal, num_pillars);
  auto reg_logits =
      triton_interface_.getOutputTensor<float>(kOutputNameReg, num_pillars, kRegressionValuesPerClass * num_classes);
  const AuxiliaryGridMapRequest& auxiliary_grid_map_request = getAuxiliaryGridMapRequest();
  if (has_auxiliary_grid_map_outputs_ && auxiliary_grid_map_request.any()) {
    const auto build_auxiliary_grid_map = [&](const char* layer) {
      return AuxiliaryGridMap{layer,
                              std::vector<float>(static_cast<std::size_t>(num_pillars)),
                              pillar_grid_.grid_x,
                              pillar_grid_.grid_y,
                              x_min_,
                              x_max_,
                              y_min_,
                              y_max_};
    };

    if (auxiliary_grid_map_request.density && auxiliary_grid_map_request.dynamic) {
      auto density_logits = triton_interface_.getOutputTensor<float>(kOutputNameDensity, num_pillars);
      auto dynamic_occupancy_logits = triton_interface_.getOutputTensor<float>(kOutputNameDynamicOccupancy, num_pillars);

      density_grid_map_ = build_auxiliary_grid_map("density");
      dynamic_grid_map_ = build_auxiliary_grid_map("dynamic");

      for (int ix = 0; ix < pillar_grid_.grid_x; ++ix) {
        for (int iy = 0; iy < pillar_grid_.grid_y; ++iy) {
          const std::size_t flat_index = static_cast<std::size_t>(ix * pillar_grid_.grid_y + iy);
          const Eigen::Index tensor_index = static_cast<Eigen::Index>(flat_index);
          density_grid_map_->values[flat_index] = densityTransform(sigmoid(density_logits(tensor_index)));
          dynamic_grid_map_->values[flat_index] = sigmoid(dynamic_occupancy_logits(tensor_index));
        }
      }
    } else if (auxiliary_grid_map_request.density) {
      auto density_logits = triton_interface_.getOutputTensor<float>(kOutputNameDensity, num_pillars);

      density_grid_map_ = build_auxiliary_grid_map("density");
      for (int ix = 0; ix < pillar_grid_.grid_x; ++ix) {
        for (int iy = 0; iy < pillar_grid_.grid_y; ++iy) {
          const std::size_t flat_index = static_cast<std::size_t>(ix * pillar_grid_.grid_y + iy);
          density_grid_map_->values[flat_index] =
              densityTransform(sigmoid(density_logits(static_cast<Eigen::Index>(flat_index))));
        }
      }
    } else if (auxiliary_grid_map_request.dynamic) {
      auto dynamic_occupancy_logits = triton_interface_.getOutputTensor<float>(kOutputNameDynamicOccupancy, num_pillars);

      dynamic_grid_map_ = build_auxiliary_grid_map("dynamic");
      for (int ix = 0; ix < pillar_grid_.grid_x; ++ix) {
        for (int iy = 0; iy < pillar_grid_.grid_y; ++iy) {
          const std::size_t flat_index = static_cast<std::size_t>(ix * pillar_grid_.grid_y + iy);
          dynamic_grid_map_->values[flat_index] = sigmoid(dynamic_occupancy_logits(static_cast<Eigen::Index>(flat_index)));
        }
      }
    }
  }

  pcod_common::PbodOutputsView view;
  view.focal_logits = focal_logits.data();
  view.size_posterior = size_posterior.data();
  view.class_logits = class_logits.data();
  view.reg_logits = reg_logits.data();
  view.num_pillars = num_pillars;
  view.num_classes = num_classes;
  view.reg_dim = kRegressionValuesPerClass;

  return pcod_common::DecodePbod(view, pillar_grid_, postprocess_config_);
}

}  // namespace point_cloud_object_detection
