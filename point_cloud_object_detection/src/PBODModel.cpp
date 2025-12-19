#include "point_cloud_object_detection/PBODModel.hpp"
#include "point_cloud_object_detection/Utils.hpp"

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
      da_fov_rad_{static_cast<float>(model_config_.detection_area_fov_deg * M_PI / 180.0)} {
  int pillar_map_size_x = model_config_.pillar_map_size[0] * model_config_.first_up_stride / model_config_.stride[0];
  int pillar_map_size_y = model_config_.pillar_map_size[1] * model_config_.first_up_stride / model_config_.stride[0];
  float half_size_height =
      (model_config_.pillar_map_range[0][1] - model_config_.pillar_map_range[0][0]) / pillar_map_size_x / 2;
  float half_size_width =
      (model_config_.pillar_map_range[1][1] - model_config_.pillar_map_range[1][0]) / pillar_map_size_y / 2;
  auto height_range = linspace(model_config_.pillar_map_range[0][0] + half_size_height,
                               model_config_.pillar_map_range[0][1] - half_size_height, pillar_map_size_x);
  auto width_range = linspace(model_config_.pillar_map_range[1][0] + half_size_width,
                              model_config_.pillar_map_range[1][1] - half_size_width, pillar_map_size_y);

  Eigen::MatrixXf heights = Eigen::Map<Eigen::MatrixXf>(height_range.data(), 1, pillar_map_size_x);
  Eigen::MatrixXf widths = Eigen::Map<Eigen::MatrixXf>(width_range.data(), pillar_map_size_y, 1);
  Eigen::MatrixXf heights_tiled = heights.replicate(pillar_map_size_y, 1);
  Eigen::MatrixXf widths_tiled = widths.replicate(1, pillar_map_size_x);

  // first flatten matrices, then merge
  Eigen::VectorXf heights_vec = Eigen::Map<Eigen::VectorXf>(heights_tiled.data(), heights_tiled.size());
  Eigen::VectorXf widths_vec = Eigen::Map<Eigen::VectorXf>(widths_tiled.data(), widths_tiled.size());
  Eigen::VectorXf z_range(heights_tiled.rows() * heights_tiled.cols());
  z_range.setZero();
  pillar_map_xyz_ = Eigen::MatrixXf(heights_tiled.rows() * heights_tiled.cols(), 3);
  pillar_map_xyz_ << heights_vec, widths_vec, z_range;
}

void PBODModel::setAdditionalPointFeatures(const float* feature_values, std::size_t point_count,
                                           std::size_t feature_stride) {
  external_point_features_ = feature_values;
  external_point_feature_count_ = point_count;
  external_point_feature_stride_ = feature_stride;
}

std::map<std::string, std::vector<int64_t>> PBODModel::getSpecialOutputShapes() {
  // As the regression output size is -1, we need to provide the real value here
  return {{output_name_reg_, {pillar_map_xyz_.rows(), 7 + 2 * model_config_.with_velocity}}};
}

void PBODModel::setupModelInput(const PointCloud& point_cloud) {
  if (model_config_.mask_is_bool) {
    setupModelInputT<bool>(point_cloud);
  } else {
    setupModelInputT<float>(point_cloud);
  }
}

std::vector<BoundingBox> PBODModel::modelOutputToBoxes() {
  auto class_logits = triton_interface_.getOutputTensor<float>(output_name_class_, pillar_map_xyz_.rows(),
                                                               model_config_.predicted_class_names.size());
  auto size_posterior = triton_interface_.getOutputTensor<float>(output_name_size_, pillar_map_xyz_.rows(), 3);
  auto focal_logits = triton_interface_.getOutputTensor<float>(output_name_focal_, pillar_map_xyz_.rows());
  auto reg_logits = triton_interface_.getOutputTensor<float>(output_name_reg_, pillar_map_xyz_.rows(),
                                                             7 + 2 * model_config_.with_velocity);

  Eigen::Vector<Eigen::Index, Eigen::Dynamic> classes{class_logits.rows()};
  for (Eigen::Index i = 0; i < class_logits.rows(); ++i) {
    class_logits.row(i).maxCoeff(&classes(i));
  }

  std::vector<BoundingBox> objects;
  for (Eigen::Index index = 0; index < focal_logits.rows(); ++index) {
    float score = 1.F / (1.F + std::exp(-focal_logits(index, 0)));
    if (score < model_config_.nms_score_threshold.at(classes(index))) {
      continue;
    }

    BoundingBox bounding_box;
    bounding_box.length = std::exp(reg_logits(index, 3)) * size_posterior(index, 0);
    bounding_box.width = std::exp(reg_logits(index, 4)) * size_posterior(index, 1);
    bounding_box.height = std::exp(reg_logits(index, 5)) * size_posterior(index, 2);
    bounding_box.center[0] = reg_logits(index, 0) * size_posterior(index, 0) + pillar_map_xyz_(index, 0);
    bounding_box.center[1] = reg_logits(index, 1) * size_posterior(index, 1) + pillar_map_xyz_(index, 1);
    bounding_box.z = reg_logits(index, 2) * size_posterior(index, 2) + pillar_map_xyz_(index, 2);
    bounding_box.yaw = wrap_to_range(reg_logits(index, 6), -M_PI, M_PI);
    if (std::isnan(bounding_box.length) || std::isnan(bounding_box.width) || std::isnan(bounding_box.height) ||
        std::isnan(bounding_box.yaw)) {
      continue;
    }
    if (model_config_.with_velocity) {
      bounding_box.has_velocity = true;
      bounding_box.v_x = reg_logits(index, 7);
      bounding_box.v_y = reg_logits(index, 8);
    }

    for (std::size_t class_idx = 0; class_idx < model_config_.predicted_class_names.size(); class_idx++) {
      // Here, only logits are stored as scores, because we don't want to waist resources by computing the softmax
      // for all the boxes that will be filtered out by NMS anyway. Softmax on the remaining ones is computed
      // later in sanitize_classifications
      bounding_box.classification.push_back({class_idx, class_logits(index, class_idx)});
    }

    bounding_box.existence_probability = score;

    objects.push_back(bounding_box);
  }
  return objects;
}

}  // namespace point_cloud_object_detection
