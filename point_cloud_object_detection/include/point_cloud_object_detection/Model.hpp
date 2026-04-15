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
};

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

  std::vector<BoundingBox> inferAndDecode(
      std::vector<std::chrono::time_point<std::chrono::high_resolution_clock>>& timestamps);

  /**
   * @brief Get the points that were actually used as input to the model after filtering
   * 
   * @return const PointCloud& Reference to the filtered input points
   */
  const PointCloud& getFilteredInputPoints() const;
  std::size_t getFilteredInputPointCount() const;

  virtual std::map<std::string, std::vector<int64_t>> getSpecialOutputShapes() { return {}; };
  virtual const std::optional<AuxiliaryGridMap>& getDensityGridMap() const { return density_grid_map_; }
  virtual const std::optional<AuxiliaryGridMap>& getOccupancyGridMap() const { return occupancy_grid_map_; }

  virtual ~Model() = default;    // Any method virtual -> destructor virtual
  Model(const Model&) = delete;  // Rule of five
  Model(Model&&) = delete;
  Model& operator=(const Model&) = delete;
  Model& operator=(Model&&) = delete;

 protected:
  triton_cpp::TritonInterface& triton_interface_;
  mutable PointCloud filtered_input_points_;  // Store points actually used as model input
  std::size_t filtered_input_point_count_ = 0;
  std::optional<AuxiliaryGridMap> density_grid_map_;
  std::optional<AuxiliaryGridMap> occupancy_grid_map_;

  virtual void setupModelInput(const PointCloud& point_cloud) = 0;
  virtual std::vector<BoundingBox> modelOutputToBoxes() = 0;
};

}  // namespace point_cloud_object_detection
