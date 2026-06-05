// Copyright Institute for Automotive Engineering (ika), RWTH Aachen University
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

#include <rclcpp/logging.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include "Definitions.hpp"
#include "Model.hpp"
#include "PointTypes.hpp"

#include "pcod_common/pbod_postprocess.hpp"
#include "pcod_common/pillar_grid.hpp"
#include "pcod_common/pillar_preprocess_cuda.hpp"
#include "pcod_common/point_preprocess.hpp"

namespace point_cloud_object_detection {

class PBODModel : public Model {
 public:
  static constexpr const char* kInputNamePointFeatures = "point_features";
  static constexpr const char* kInputNamePillarIds = "pillar_ids";
  static constexpr const char* kInputNameValidMask = "valid_mask";
  static constexpr const char* kInputNamePillarMasks = "pillar_masks";
  static constexpr const char* kInputNamePillarIndices = "pillar_indices";

  static constexpr const char* kOutputNameFocal = "focal_logits";
  static constexpr const char* kOutputNameReg = "reg_logits";
  static constexpr const char* kOutputNameClass = "class_logits";
  static constexpr const char* kOutputNameSize = "size_posterior";
  static constexpr const char* kOutputNameDensity = "density_logits";
  static constexpr const char* kOutputNameOccupancy = "occupancy_logits";

  static constexpr int kPreprocessedFeatureDim = 18;
  static constexpr int kPillarIndexDim = 2;
  static constexpr int kPointCoordinateDim = 3;
  static constexpr int kRegressionValuesPerClass = 7;
  static constexpr int kSizeValuesPerClass = 3;
  enum FeatureIndex : int {
    kFeatureX = 0,
    kFeatureY,
    kFeatureZ,
    kFeatureRadius,
    kFeatureZRelative,
    kFeatureInverseRadius,
    kFeatureSinTheta,
    kFeatureCosTheta,
    kFeatureVarianceX,
    kFeatureVarianceY,
    kFeatureVarianceZ,
    kFeatureIntensity,
    kFeatureClusterOffsetX,
    kFeatureClusterOffsetY,
    kFeatureClusterOffsetZ,
    kFeatureCenterOffsetX,
    kFeatureCenterOffsetY,
    kFeatureCenterOffsetZ
  };

  static constexpr std::array<const char*, 5> kExpectedInputNames{kInputNamePointFeatures, kInputNamePillarIds,
                                                                  kInputNameValidMask, kInputNamePillarMasks,
                                                                  kInputNamePillarIndices};
  static constexpr std::array<const char*, 6> kExpectedOutputNames{
      kOutputNameFocal, kOutputNameReg, kOutputNameClass, kOutputNameSize, kOutputNameDensity, kOutputNameOccupancy};

  /**
   * @brief Validate that the loaded Triton model exposes the expected PBOD input and output tensors.
   */
  static void validateInterface(const triton_cpp::TritonInterface& triton_interface);

  /**
   * @brief Construct a PBOD model wrapper using Triton and manifest-derived runtime configuration.
   */
  PBODModel(triton_cpp::TritonInterface& triton_interface, const ModelConfig& model_config);

  /**
   * @brief Return shape overrides for optional PBOD auxiliary output tensors.
   */
  std::map<std::string, std::vector<int64_t>> getSpecialOutputShapes() override;
  /**
   * @brief Build model input tensors directly from a ROS PointCloud2 message.
   */
  void prepareModelInputFromPointCloud2(const sensor_msgs::msg::PointCloud2& point_cloud_msg, uint32_t x_offset,
                                        uint32_t y_offset, uint32_t z_offset, uint32_t feature_offset,
                                        uint8_t feature_datatype, bool needs_swap, bool materialize_filtered_points);
  /**
   * @brief Destroy the PBOD model wrapper.
   */
  ~PBODModel() override = default;
  /**
   * @brief PBOD model wrappers own tensor buffers and cannot be copied.
   */
  PBODModel(const PBODModel&) = delete;  // Rule of five
  /**
   * @brief PBOD model wrappers own tensor buffers and cannot be copy-assigned.
   */
  PBODModel& operator=(const PBODModel&) = delete;
  /**
   * @brief PBOD model wrappers own tensor buffers and cannot be moved.
   */
  PBODModel(PBODModel&&) = delete;
  /**
   * @brief PBOD model wrappers own tensor buffers and cannot be move-assigned.
   */
  PBODModel& operator=(PBODModel&&) = delete;

 protected:
  /**
   * @brief Convert a PCL point cloud into PBOD input tensors.
   */
  void setupModelInput(const PointCloud& point_cloud) override;
  /**
   * @brief Decode PBOD output tensors into bounding boxes and auxiliary grids.
   */
  std::vector<BoundingBox> modelOutputToBoxes() override;

 private:
  struct TensorResetState {
    std::vector<int> active_point_rows;
    std::vector<int> active_pillar_ids;
    bool initialized = false;
  };

  /**
   * @brief Fill model input tensors from an abstract point accessor.
   */
  template <typename PointGetter>
  void setupModelInputFromGetter(int n_points, PointGetter&& get_point, bool materialize_filtered_points);
  /**
   * @brief Populate PBOD input tensors using the CPU preprocessing path.
   */
  void populateModelInputOnCpu(float* point_features, std::int64_t* pillar_ids, bool* valid_mask, bool* pillar_masks,
                               int num_selected);
  /**
   * @brief Populate PBOD input tensors using CUDA preprocessing with host-visible tensor buffers.
   */
  bool populateModelInputOnGpu(float* point_features, std::int64_t* pillar_ids, bool* valid_mask, bool* pillar_masks,
                               int num_selected);
  /**
   * @brief Populate PBOD input tensors using CUDA preprocessing directly into device tensor buffers.
   */
  bool populateModelInputOnGpuToDevice(float* point_features, std::int64_t* pillar_ids, bool* valid_mask,
                                       bool* pillar_masks, int num_selected);

  const ModelConfig model_config_;
  pcod_common::PillarGrid pillar_grid_;
  pcod_common::PbodPostprocessConfig postprocess_config_;

  // Cached range values for performance
  const float x_min_;
  const float x_max_;
  const float y_min_;
  const float y_max_;
  const float z_min_;
  const float z_max_;
  const float voxel_x_;
  const float voxel_y_;
  const float voxel_z_;
  const pcod_common::PointFeatureNormalizationType normalization_type_;
  const float value_threshold_;
  const float min_value_;
  const float max_value_;
  const float norm_epsilon_;
  const int max_num_points_;
  const int preprocessed_feature_dim_;
  const int num_pillars_;
  const std::int64_t pillar_id_sentinel_;
  std::vector<int64_t> pillar_indices_;
  bool pillar_indices_initialized_ = false;
  TensorResetState tensor_reset_state_;
  std::vector<int32_t> pillar_counts_;
  std::vector<float> pillar_sum_;
  std::vector<float> pillar_sq_sum_;
  std::vector<float> pillar_mean_;
  std::vector<float> pillar_var_;
  std::vector<int> pillar_ids_raw_;
  std::vector<int> pillar_ix_raw_;
  std::vector<int> pillar_iy_raw_;
  std::vector<int> active_pillar_ids_scratch_;
  std::vector<pcod_common::PillarPreprocessPoint> selected_point_records_;
  // No-detection zone point filtering
  const bool remove_points_in_zone_;
  const float nd_x_min_;
  const float nd_x_max_;
  const float nd_y_min_;
  const float nd_y_max_;

  // Detection area (sector) point filtering
  const bool det_area_remove_outside_;
  const float da_cx_;
  const float da_cy_;
  const float da_radius_;
  const float da_bearing_rad_;
  const float da_fov_rad_;
  pcod_common::PointPreprocessor point_preprocessor_;
  pcod_common::PillarPreprocessCudaContext cuda_preprocess_context_;
  bool has_auxiliary_grid_map_outputs_ = false;
};

}  // namespace point_cloud_object_detection
