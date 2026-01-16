#include "point_cloud_object_detection/PBODModel.hpp"

namespace point_cloud_object_detection {
PBODModel::PBODModel(triton_cpp::TritonInterface& triton_interface, ModelConfig& model_config)
    : Model(triton_interface),
      model_config_{model_config},
      input_name_xyz_{SAVED_MODEL_INPUT_NAME_XYZ},
      input_name_feature_{SAVED_MODEL_INPUT_NAME_FEATURE},
      input_name_mask_{SAVED_MODEL_INPUT_NAME_MASK},
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
      intensity_threshold_{static_cast<float>(model_config_.intensity_threshold)},
      zero_intensity_{model_config_.zero_intensity},
      max_num_points_{model_config_.max_num_points},
      num_point_features_{model_config_.num_point_features},
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
                           intensity_threshold_,
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
  pillar_grid_ = pcod_common::BuildPillarGrid(
      {static_cast<int>(model_config_.pillar_map_size[0]), static_cast<int>(model_config_.pillar_map_size[1])},
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
  // As the regression output size is -1, we need to provide the real value here
  const int64_t num_pillars = static_cast<int64_t>(pillar_grid_.grid_x * pillar_grid_.grid_y);
  return {{output_name_reg_, {num_pillars, 7}}};
}

void PBODModel::setupModelInput(const PointCloud& point_cloud) {
  if (model_config_.mask_is_bool) {
    setupModelInputT<bool>(point_cloud);
  } else {
    setupModelInputT<float>(point_cloud);
  }
}

std::vector<BoundingBox> PBODModel::modelOutputToBoxes() {
  const int num_pillars = pillar_grid_.grid_x * pillar_grid_.grid_y;
  const int num_classes = static_cast<int>(model_config_.predicted_class_names.size());
  auto class_logits = triton_interface_.getOutputTensor<float>(output_name_class_, num_pillars, num_classes);
  auto size_posterior = triton_interface_.getOutputTensor<float>(output_name_size_, num_pillars, 3);
  auto focal_logits = triton_interface_.getOutputTensor<float>(output_name_focal_, num_pillars);
  auto reg_logits = triton_interface_.getOutputTensor<float>(output_name_reg_, num_pillars, 7);

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
