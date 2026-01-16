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
#include "pcod_common/point_preprocess.hpp"
#include "pcod_common/pillar_grid.hpp"

namespace point_cloud_object_detection {

class PBODModel : public Model {
 public:
  PBODModel(triton_cpp::TritonInterface &triton_interface, ModelConfig &model_config);

  const std::string SAVED_MODEL_INPUT_NAME_XYZ = "points_xyz";
  const std::string SAVED_MODEL_INPUT_NAME_FEATURE = "points_feature";
  const std::string SAVED_MODEL_INPUT_NAME_MASK = "points_mask";

  const std::string SAVED_MODEL_OUTPUT_NAME_FOCAL = "focal_logits";
  const std::string SAVED_MODEL_OUTPUT_NAME_REG = "reg_logits";
  const std::string SAVED_MODEL_OUTPUT_NAME_CLASS = "class_logits";
  const std::string SAVED_MODEL_OUTPUT_NAME_SIZE = "size_posterior";

  virtual std::map<std::string, std::vector<int64_t>> getSpecialOutputShapes() override;
  void setAdditionalPointFeatures(const float *feature_values, std::size_t point_count,
                                  std::size_t feature_stride) override;

  virtual ~PBODModel() = default;         // Any method virtual -> destructor virtual
  PBODModel(const PBODModel &) = delete;  // Rule of five
  PBODModel &operator=(const PBODModel &) = delete;
  PBODModel(PBODModel &&) = delete;
  PBODModel &operator=(PBODModel &&) = delete;

 protected:
  virtual void setupModelInput(const PointCloud &point_cloud) override;
 virtual std::vector<BoundingBox> modelOutputToBoxes() override;

 private:
  ModelConfig &model_config_;
  pcod_common::PillarGrid pillar_grid_;

  const std::string input_name_xyz_;
  const std::string input_name_feature_;
  const std::string input_name_mask_;

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
  const float intensity_threshold_;
  const bool zero_intensity_;
  const int max_num_points_;
  const int num_point_features_;
  const float *external_point_features_ = nullptr;
  std::size_t external_point_feature_stride_ = 0;
  std::size_t external_point_feature_count_ = 0;

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


  inline const float *getExtraFeatures(std::size_t point_index) const {
    if (external_point_features_ == nullptr || external_point_feature_stride_ == 0 ||
        point_index >= external_point_feature_count_) {
      return nullptr;
    }
    return external_point_features_ + point_index * external_point_feature_stride_;
  }

  template <typename FeatureMapType>
  inline void populateFeatureChannels(int tensor_idx, const Point &point, const float *extra_features,
                                      FeatureMapType &points_feature_map) const {
    points_feature_map(tensor_idx, 0) = point_preprocessor_.NormalizeIntensity(point.intensity);
    if (num_point_features_ <= 1) {
      return;
    }

    for (int feature_idx = 1; feature_idx < num_point_features_; ++feature_idx) {
      float value = 0.0f;
      if (extra_features != nullptr) {
        const std::size_t offset = static_cast<std::size_t>(feature_idx - 1);
        value = extra_features[offset];
      }
      points_feature_map(tensor_idx, feature_idx) = value;
    }
  }

  // Helper function to set point data in tensors
  template <typename MaskType, typename XYZMapType, typename FeatureMapType, typename MaskMapType>
  inline void setPointData(int tensor_idx, const Point &point, const float *extra_features, XYZMapType &points_xyz_map,
                           FeatureMapType &points_feature_map, MaskMapType &points_mask_map) const {
    points_xyz_map(tensor_idx, 0) = point.x;
    points_xyz_map(tensor_idx, 1) = point.y;
    points_xyz_map(tensor_idx, 2) = point.z;
    populateFeatureChannels(tensor_idx, point, extra_features, points_feature_map);
    points_mask_map(tensor_idx) = static_cast<MaskType>(1);
  }

  // Setup model input tensors from point cloud data
  template <typename MaskType>
  void setupModelInputT(const PointCloud &point_cloud) {
    auto points_xyz_map = triton_interface_.getInputTensor<float>(input_name_xyz_, max_num_points_, 3);
    auto points_feature_map =
        triton_interface_.getInputTensor<float>(input_name_feature_, max_num_points_, num_point_features_);
    auto points_mask_map = triton_interface_.getInputTensor<MaskType>(input_name_mask_, max_num_points_);
    points_xyz_map.setZero();
    points_feature_map.setZero();
    points_mask_map.setZero();

    // Clear and reserve space for filtered points
    filtered_input_points_.clear();
    filtered_input_points_.reserve(std::min(static_cast<int>(point_cloud.size()), max_num_points_));

    int n_cloud_points = point_cloud.size();
    if (n_cloud_points == 0) {
      RCLCPP_DEBUG(rclcpp::get_logger("PBODModel"), "Input point cloud is empty.");
      return;
    }

    // Use thread-local static generator for better performance
    static thread_local std::mt19937 gen(std::random_device{}());

    // Process all points using reservoir sampling algorithm
    // This naturally handles both cases:
    // total points < max_num_points_ and total points >= max_num_points_
    int valid_point_count = 0;  // Track number of valid points processed

    for (int i = 0; i < n_cloud_points; ++i) {
      const auto &point = point_cloud[i];

      if (!point_preprocessor_.IsPointValid(point.x, point.y, point.z)) {
        continue;
      }

      valid_point_count++;
      const float *extra_features = getExtraFeatures(static_cast<std::size_t>(i));
      if (valid_point_count <= max_num_points_) {
        // Always take the first max_num_points_ valid points
        int idx = valid_point_count - 1;  // 0-based index for tensor
        setPointData<MaskType>(idx, point, extra_features, points_xyz_map, points_feature_map, points_mask_map);
        filtered_input_points_.push_back(point);
      } else {
        // Reservoir sampling: randomly decide whether to replace
        std::uniform_int_distribution<int> dist(1, valid_point_count);
        int random_idx = dist(gen);

        // Replace if random number falls within reservoir size
        if (random_idx <= max_num_points_) {
          int replace_idx = random_idx - 1;  // Convert to 0-based index
          setPointData<MaskType>(replace_idx, point, extra_features, points_xyz_map, points_feature_map,
                                 points_mask_map);
          filtered_input_points_[replace_idx] = point;  // Replace the point in filtered list too
        }
      }
    }
    RCLCPP_DEBUG(rclcpp::get_logger("PBODModel"), "Filtered %d valid points for model input",
                 valid_point_count < max_num_points_ ? valid_point_count : max_num_points_);
  }
};

}  // namespace point_cloud_object_detection

#endif  // POINT_CLOUD_OBJECT_DETECTION__PBOD_MODEL_HPP_
