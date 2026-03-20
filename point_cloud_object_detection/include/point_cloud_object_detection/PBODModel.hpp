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

#include "Definitions.hpp"
#include "Model.hpp"
#include "PointTypes.hpp"

#include "pcod_common/pbod_postprocess.hpp"
#include "pcod_common/pillar_grid.hpp"
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
  static constexpr std::array<const char*, 4> kExpectedOutputNames{kOutputNameFocal, kOutputNameReg, kOutputNameClass,
                                                                   kOutputNameSize};

  static void validateInterface(const triton_cpp::TritonInterface& triton_interface);

  PBODModel(triton_cpp::TritonInterface& triton_interface, ModelConfig& model_config);

  std::map<std::string, std::vector<int64_t>> getSpecialOutputShapes() override;
  ~PBODModel() override = default;
  PBODModel(const PBODModel&) = delete;  // Rule of five
  PBODModel& operator=(const PBODModel&) = delete;
  PBODModel(PBODModel&&) = delete;
  PBODModel& operator=(PBODModel&&) = delete;

 protected:
  void setupModelInput(const PointCloud& point_cloud) override;
  std::vector<BoundingBox> modelOutputToBoxes() override;

 private:
  ModelConfig& model_config_;
  pcod_common::PillarGrid pillar_grid_;

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
  const float intensity_threshold_;
  const float min_intensity_;
  const float max_intensity_;
  const float norm_epsilon_;
  const bool zero_intensity_;
  const int max_num_points_;
  const int preprocessed_feature_dim_;
  const int num_pillars_;
  std::vector<int64_t> pillar_indices_;
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
};

}  // namespace point_cloud_object_detection
