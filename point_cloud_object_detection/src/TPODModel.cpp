#include <tensorflow/cc/ops/math_ops.h>
#include <tensorflow/core/framework/tensor.h>
#include "point_cloud_object_detection/Model.hpp"
#include "point_cloud_object_detection/Utils.hpp"

namespace point_cloud_object_detection {
TPODModel::TPODModel(ModelConfig& model_config, bool is_frozen_graph)
    : Model(is_frozen_graph),
      model_config_{model_config},
      input_name_xyz_{is_frozen_graph ? FROZEN_GRAPH_INPUT_NAME_XYZ : SAVED_MODEL_INPUT_NAME_XYZ},
      input_name_feature_{is_frozen_graph ? FROZEN_GRAPH_INPUT_NAME_FEATURE : SAVED_MODEL_INPUT_NAME_FEATURE},
      input_name_mask_{is_frozen_graph ? FROZEN_GRAPH_INPUT_NAME_MASK : SAVED_MODEL_INPUT_NAME_MASK},
      output_name_cls_{is_frozen_graph ? FROZEN_GRAPH_OUTPUT_NAME_CLS : SAVED_MODEL_OUTPUT_NAME_CLS},
      output_name_reg_{is_frozen_graph ? FROZEN_GRAPH_OUTPUT_NAME_REG : SAVED_MODEL_OUTPUT_NAME_REG} {
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

SavedModelInputType TPODModel::getZeroedModelInputTensors() {
  SavedModelInputType model_inputs{
      {{input_name_xyz_,
        tensorflow::Tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({1, model_config_.max_num_points, 3}))},
       {input_name_feature_,
        tensorflow::Tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({1, model_config_.max_num_points, 1}))},
       {input_name_mask_,
        tensorflow::Tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({1, model_config_.max_num_points}))}}};
  Eigen::TensorMap<Eigen::Tensor<float, 3, Eigen::RowMajor, int64_t>, Eigen::Aligned, Eigen::MakePointer>
      points_xyz_map = model_inputs[0].second.tensor<float, 3>();
  Eigen::TensorMap<Eigen::Tensor<float, 3, Eigen::RowMajor, int64_t>, Eigen::Aligned, Eigen::MakePointer>
      points_feature_map = model_inputs[1].second.tensor<float, 3>();
  Eigen::TensorMap<Eigen::Tensor<float, 2, Eigen::RowMajor, int64_t>, Eigen::Aligned, Eigen::MakePointer>
      points_mask_map = model_inputs[2].second.tensor<float, 2>();
  points_xyz_map.setZero();
  points_feature_map.setZero();
  points_mask_map.setZero();
  return model_inputs;
}

std::vector<std::string> TPODModel::getOutputNames() {
  std::vector<std::string> output_names{output_name_cls_, output_name_reg_};
  return output_names;
}

void TPODModel::pointCloudToModelInput(const PointCloud& point_cloud, SavedModelInputType& model_inputs) {
  Eigen::TensorMap<Eigen::Tensor<float, 3, Eigen::RowMajor, int64_t>, Eigen::Aligned, Eigen::MakePointer>
      points_xyz_map = model_inputs[0].second.tensor<float, 3>();
  Eigen::TensorMap<Eigen::Tensor<float, 3, Eigen::RowMajor, int64_t>, Eigen::Aligned, Eigen::MakePointer>
      points_feature_map = model_inputs[1].second.tensor<float, 3>();
  Eigen::TensorMap<Eigen::Tensor<float, 2, Eigen::RowMajor, int64_t>, Eigen::Aligned, Eigen::MakePointer>
      points_mask_map = model_inputs[2].second.tensor<float, 2>();

  int n_cloud_points = point_cloud.size();

  int pcl_index = 0;
  int tensor_index = 0;
  // TODO(TBD): Avoid this loop. Possible??
  while (pcl_index < n_cloud_points && tensor_index < model_config_.max_num_points) {
    const auto& point = point_cloud[pcl_index];
    // filter points to detection range (and filter out NaNs)
    if ((point.x < model_config_.pillar_map_range[0][0]) || (point.x >= model_config_.pillar_map_range[0][1]) ||
        std::isnan(point.x) || (point.y < model_config_.pillar_map_range[1][0]) ||
        (point.y >= model_config_.pillar_map_range[1][1]) || std::isnan(point.y) ||
        (point.z < model_config_.pillar_map_range[2][0]) || (point.z >= model_config_.pillar_map_range[2][1]) ||
        std::isnan(point.z)) {
      ++pcl_index;
      continue;
    }
    points_xyz_map(0, tensor_index, 0) = point.x;
    points_xyz_map(0, tensor_index, 1) = point.y;
    points_xyz_map(0, tensor_index, 2) = point.z;
    points_feature_map(0, tensor_index, 0) = std::min(1.F, point.intensity / model_config_.intensity_threshold);
    points_mask_map(0, tensor_index) = 1.F;
    ++pcl_index;
    ++tensor_index;
  }
}

std::vector<BoundingBox> TPODModel::modelOutputToBoxes(const SavedModelOutputType& model_output) {
  std::vector<BoundingBox> objects;
  Eigen::TensorMap<Eigen::Tensor<const float, 3, Eigen::RowMajor, int64_t>, Eigen::Aligned, Eigen::MakePointer>
      cls_logits = model_output.at(output_name_cls_).tensor<float, 3>();
  Eigen::TensorMap<Eigen::Tensor<const float, 3, Eigen::RowMajor, int64_t>, Eigen::Aligned, Eigen::MakePointer>
      reg_logits = model_output.at(output_name_reg_).tensor<float, 3>();
  // auto& class_logits = model_output.at(output_name_cls_);
  // const float cls_threshold_inv_sigmoid = std::log(model_config_.cls_threshold / (1.F - model_config_.cls_threshold));

  Eigen::Tensor<int64_t, 2, Eigen::RowMajor, int64_t> classes = cls_logits.argmax(2);

  for (int index = 0; index < cls_logits.dimension(1); ++index) {
    float score = 1.F / (1.F + std::exp(-cls_logits(0, index, classes(0, index))));
    if (score < model_config_.cls_threshold) {
      continue;
    }

    BoundingBox bounding_box;
    bounding_box.length = std::exp(reg_logits(0, index, 3));
    bounding_box.width = std::exp(reg_logits(0, index, 4));
    bounding_box.height = std::exp(reg_logits(0, index, 5));
    bounding_box.center(0) = reg_logits(0, index, 0);
    bounding_box.center(1) = reg_logits(0, index, 1);
    bounding_box.z = reg_logits(0, index, 2);
    bounding_box.yaw = std::atan2(reg_logits(0, index, 6), reg_logits(0, index, 7));
    if (std::isnan(bounding_box.length) || std::isnan(bounding_box.width) || std::isnan(bounding_box.height) ||
        std::isnan(bounding_box.yaw)) {
      continue;
    }
    if (model_config_.with_velocity) {
      bounding_box.has_velocity = true;
      bounding_box.v_x = reg_logits(0, index, 8);
      bounding_box.v_y = reg_logits(0, index, 9);
    }

    for (size_t class_idx = 0; class_idx < model_config_.predicted_class_names.size(); class_idx++) {
      // For TPOD, sigmoid(cls_logit) is scored and will be normalized to add up to 1 when convertig to perception_msgs
      bounding_box.classification.push_back({class_idx, 1.F / (1.F + std::exp(-cls_logits(0, index, class_idx)))});
    }

    bounding_box.existence_probability = score;

    objects.push_back(bounding_box);
  }
  return objects;
}

}  // namespace point_cloud_object_detection
