// Copyright Institute for Automotive Engineering (ika), RWTH Aachen University
// SPDX-License-Identifier: Apache-2.0

#include "point_cloud_object_detection/Model.hpp"

namespace point_cloud_object_detection {
Model::Model(triton_cpp::TritonInterface& triton_interface) : triton_interface_{triton_interface} {}

std::vector<BoundingBox> Model::operator()(const PointCloud& point_cloud,
                                           std::vector<std::chrono::time_point<std::chrono::high_resolution_clock>>& timestamps) {
  // input Tensor creation
  setupModelInput(point_cloud);
  timestamps.push_back(std::chrono::high_resolution_clock::now());  // after input tensor creation, before inference

  return inferAndDecode(timestamps);
}

std::vector<BoundingBox> Model::inferAndDecode(
    std::vector<std::chrono::time_point<std::chrono::high_resolution_clock>>& timestamps) {
  density_grid_map_.reset();
  dynamic_grid_map_.reset();

  // inference
  triton_interface_.infer();
  timestamps.push_back(std::chrono::high_resolution_clock::now());  // after inference

  // modelOutputToBoxes
  std::vector<BoundingBox> center_boxes = modelOutputToBoxes();
  return center_boxes;
}

const PointCloud& Model::getFilteredInputPoints() const { return filtered_input_points_; }

std::size_t Model::getFilteredInputPointCount() const { return filtered_input_point_count_; }

}  // namespace point_cloud_object_detection
