// Copyright Institute for Automotive Engineering (ika), RWTH Aachen University
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pcl/point_cloud.h>
#include <triton_cpp/triton_interface.hpp>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <iostream>
#include <numeric>
#include <optional>
#include <random>
#include <string>
#include <vector>

#include "point_cloud_object_detection/Definitions.hpp"
#include "point_cloud_object_detection/PointTypes.hpp"
#include "point_cloud_object_detection/Utils.hpp"

namespace point_cloud_object_detection {
// namespace acronyms
namespace tc = triton::client;

struct AuxiliaryGridMap {
  std::string layer;
  std::vector<float> values;
  int grid_x = 0;
  int grid_y = 0;
  // Spatial range of the pillar grid in the inference frame (metres).
  // Populated by the model at inference time so downstream code does not
  // need to consult ModelConfig independently.
  float x_min = 0.0F;
  float x_max = 0.0F;
  float y_min = 0.0F;
  float y_max = 0.0F;
};

struct AuxiliaryGridMapRequest {
  bool density = false;
  bool occupancy = false;

  /**
   * @brief Return true when at least one auxiliary grid map should be decoded.
   */
  [[nodiscard]] bool any() const { return density || occupancy; }
};

/**
 * @brief Virtual class representing any model architecture's pre- and
 * post-processing
 *
 */
class Model {
 public:
  /**
   * @brief Construct a new Model object
   *
   */
  explicit Model(triton_cpp::TritonInterface& triton_interface);

  /**
   * @brief Execution of point cloud object detection including input tensor
   * creation, inference and creation of bounding box vector from model output
   *
   * @param model                                 Tensorflow model
   * @param point_cloud                           Point cloud data as model
   * input
   * @param timestamps                            Vector with instants of time
   * regarding different steps of the prediction. This function will add two
   * time points: before and after the inference
   * @return std::vector<BoundingBox>       Model output: Vector with predicted
   * bounding boxes
   */
  std::vector<BoundingBox> operator()(const PointCloud& point_cloud,
                                      std::vector<std::chrono::time_point<std::chrono::high_resolution_clock>>& timestamps);

  /**
   * @brief Run model inference on the already prepared input tensors and
   * decode the raw outputs.
   *
   * @param timestamps Timing trace that receives entries before and after
   * Triton inference.
   * @return Decoded bounding boxes.
   */
  std::vector<BoundingBox> inferAndDecode(std::vector<std::chrono::time_point<std::chrono::high_resolution_clock>>& timestamps);

  /**
   * @brief Get the points that were actually used as input to the model after
   * filtering
   *
   * @return const PointCloud& Reference to the filtered input points
   */
  const PointCloud& getFilteredInputPoints() const;
  /**
   * @brief Return the number of points kept after model input filtering.
   */
  std::size_t getFilteredInputPointCount() const;
  /**
   * @brief Configure which optional auxiliary grid-map outputs should be
   * decoded.
   */
  void setAuxiliaryGridMapRequest(const AuxiliaryGridMapRequest& request) { auxiliary_grid_map_request_ = request; }
  /**
   * @brief Return the current auxiliary grid-map output request.
   */
  const AuxiliaryGridMapRequest& getAuxiliaryGridMapRequest() const { return auxiliary_grid_map_request_; }

  /**
   * @brief Return output tensor shape overrides for model outputs with
   * runtime-dependent shapes.
   */
  virtual std::map<std::string, std::vector<int64_t>> getSpecialOutputShapes() { return {}; };
  /**
   * @brief Return the latest decoded density grid map if one was requested and
   * produced.
   */
  virtual const std::optional<AuxiliaryGridMap>& getDensityGridMap() const { return density_grid_map_; }
  /**
   * @brief Return the latest decoded occupancy grid map if one was requested
   * and produced.
   */
  virtual const std::optional<AuxiliaryGridMap>& getOccupancyGridMap() const { return occupancy_grid_map_; }

  /**
   * @brief Destroy the model interface.
   */
  virtual ~Model() = default;  // Any method virtual -> destructor virtual
  /**
   * @brief Models own Triton tensor state and cannot be copied.
   */
  Model(const Model&) = delete;  // Rule of five
  /**
   * @brief Models own Triton tensor state and cannot be moved.
   */
  Model(Model&&) = delete;
  /**
   * @brief Models own Triton tensor state and cannot be copy-assigned.
   */
  Model& operator=(const Model&) = delete;
  /**
   * @brief Models own Triton tensor state and cannot be move-assigned.
   */
  Model& operator=(Model&&) = delete;

 protected:
  triton_cpp::TritonInterface& triton_interface_;
  mutable PointCloud filtered_input_points_;  // Store points actually used as model input
  std::size_t filtered_input_point_count_ = 0;
  AuxiliaryGridMapRequest auxiliary_grid_map_request_;
  std::optional<AuxiliaryGridMap> density_grid_map_;
  std::optional<AuxiliaryGridMap> occupancy_grid_map_;

  /**
   * @brief Convert a PCL point cloud into model input tensors.
   */
  virtual void setupModelInput(const PointCloud& point_cloud) = 0;
  /**
   * @brief Decode raw model output tensors into bounding boxes.
   */
  virtual std::vector<BoundingBox> modelOutputToBoxes() = 0;
};

}  // namespace point_cloud_object_detection
