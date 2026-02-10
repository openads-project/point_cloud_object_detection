#ifndef POINT_CLOUD_OBJECT_DETECTION__PBOD_MODEL_HPP_
#define POINT_CLOUD_OBJECT_DETECTION__PBOD_MODEL_HPP_

#include <algorithm>
#include <cmath>
#include <cstddef>
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
  PBODModel(triton_cpp::TritonInterface& triton_interface, ModelConfig& model_config);

  const std::string SAVED_MODEL_INPUT_NAME_POINT_FEATURES = "point_features";
  const std::string SAVED_MODEL_INPUT_NAME_PILLAR_IDS = "pillar_ids";
  const std::string SAVED_MODEL_INPUT_NAME_VALID_MASK = "valid_mask";
  const std::string SAVED_MODEL_INPUT_NAME_PILLAR_MASKS = "pillar_masks";
  const std::string SAVED_MODEL_INPUT_NAME_PILLAR_INDICES = "pillar_indices";

  const std::string SAVED_MODEL_OUTPUT_NAME_FOCAL = "focal_logits";
  const std::string SAVED_MODEL_OUTPUT_NAME_REG = "reg_logits";
  const std::string SAVED_MODEL_OUTPUT_NAME_CLASS = "class_logits";
  const std::string SAVED_MODEL_OUTPUT_NAME_SIZE = "size_posterior";

  virtual std::map<std::string, std::vector<int64_t>> getSpecialOutputShapes() override;
  virtual ~PBODModel() = default;        // Any method virtual -> destructor virtual
  PBODModel(const PBODModel&) = delete;  // Rule of five
  PBODModel& operator=(const PBODModel&) = delete;
  PBODModel(PBODModel&&) = delete;
  PBODModel& operator=(PBODModel&&) = delete;

 protected:
  virtual void setupModelInput(const PointCloud& point_cloud) override;
  virtual std::vector<BoundingBox> modelOutputToBoxes() override;

 private:
  ModelConfig& model_config_;
  pcod_common::PillarGrid pillar_grid_;

  const std::string input_name_point_features_;
  const std::string input_name_pillar_ids_;
  const std::string input_name_valid_mask_;
  const std::string input_name_pillar_masks_;
  const std::string input_name_pillar_indices_;

  const std::string output_name_focal_;
  const std::string output_name_reg_;
  const std::string output_name_class_;
  const std::string output_name_size_;

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

#endif  // POINT_CLOUD_OBJECT_DETECTION__PBOD_MODEL_HPP_
