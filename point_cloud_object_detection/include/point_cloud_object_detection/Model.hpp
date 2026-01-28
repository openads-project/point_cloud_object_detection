#pragma once

#include <pcl/point_cloud.h>
#include <triton_cpp/triton_interface.hpp>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

#include "point_cloud_object_detection/Definitions.hpp"
#include "point_cloud_object_detection/PointTypes.hpp"
#include "point_cloud_object_detection/Utils.hpp"

namespace point_cloud_object_detection {
// namespace acronyms
namespace tc = triton::client;

/**
 * @brief Virtual class representing any model architecture's pre- and post-processing
 *
 */
class Model {
 public:
  /**
   * @brief Construct a new Model object
   *
   */
  Model(triton_cpp::TritonInterface& triton_interface);

  /**
   * @brief Execution of point cloud object detection including input tensor creation, inference and creation of bounding box vector from model output
   *
   * @param model                                 Tensorflow model
   * @param point_cloud                           Point cloud data as model input
   * @param timestamps                            Vector with instants of time regarding different steps of the prediction. This function will add two time points: before and after the inference
   * @return std::vector<BoundingBox>       Model output: Vector with predicted bounding boxes
   */
  std::vector<BoundingBox> operator()(
      const PointCloud& point_cloud,
      std::vector<std::chrono::time_point<std::chrono::high_resolution_clock>>& timestamps);

  /**
   * @brief Get the points that were actually used as input to the model after filtering
   * 
   * @return const PointCloud& Reference to the filtered input points
   */
  const PointCloud& getFilteredInputPoints() const;

  /**
   * @brief Supply additional per-point feature channels (excluding XYZ and intensity) for the next inference call.
   * @param feature_values Pointer to contiguous feature storage (row-major) or nullptr if unavailable.
   * @param point_count Number of points represented in the buffer.
   * @param feature_stride Number of additional feature values stored per point.
   */
  virtual void setAdditionalPointFeatures(const float* /*feature_values*/, std::size_t /*point_count*/,
                                          std::size_t /*feature_stride*/) {}

  virtual std::map<std::string, std::vector<int64_t>> getSpecialOutputShapes() { return {}; };

  virtual ~Model() = default;    // Any method virtual -> destructor virtual
  Model(const Model&) = delete;  // Rule of five
  Model(Model&&) = delete;
  Model& operator=(const Model&) = delete;
  Model& operator=(Model&&) = delete;

 protected:
  triton_cpp::TritonInterface& triton_interface_;
  mutable PointCloud filtered_input_points_;  // Store points actually used as model input

  virtual void setupModelInput(const PointCloud& point_cloud) = 0;
  virtual std::vector<BoundingBox> modelOutputToBoxes() = 0;
};

}  // namespace point_cloud_object_detection
