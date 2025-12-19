#pragma once

#include <string>
#include <vector>

#include "point_cloud_object_detection/Definitions.hpp"

namespace point_cloud_object_detection {
class NonMaxSuppression {
 public:
  NonMaxSuppression(ModelConfig& model_config, Params& params);

  /**
   * @brief Non maximum suppression discarding bounding boxes according to iou and
   * score thresholds
   *
   * @param bounding_boxes_center     Vector with bounding boxes
   *
   * @return                          Vector with bounding boxes after NMS
   */
  void nms(std::vector<BoundingBox>& bboxes);

  static const float kInternalScoreThreshold;

 private:
  // parameters
  ModelConfig& model_config_;
  Params& params_;
};

}  // namespace point_cloud_object_detection
