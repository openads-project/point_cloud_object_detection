// Copyright Institute for Automotive Engineering (ika), RWTH Aachen University
// SPDX-License-Identifier: Apache-2.0

#include "point_cloud_object_detection/Utils.hpp"

#include <array>
#include <cmath>
#include <stdexcept>

#include "pcod_common/nms.hpp"

namespace {

void expectNear(double actual, double expected) {
  if (!std::isfinite(actual) || std::abs(actual - expected) > 1e-6) {
    throw std::runtime_error("Merged ROS classification probability differs from the weighted mixture");
  }
}

}  // namespace

/** Verify that merged log probabilities survive the existing ROS message conversion. */
int main() {
  pcod_common::BoundingBox a;
  a.center = {0.0F, 0.0F};
  a.length = 10.0F;
  a.width = 2.5F;
  a.height = 3.0F;
  a.detection_score = 0.9F;
  a.classification = {{0, std::log(0.8F)}, {1, std::log(0.15F)}, {2, std::log(0.05F)}};
  auto b = a;
  b.detection_score = 0.6F;
  b.classification = {{0, std::log(0.1F)}, {1, std::log(0.85F)}, {2, std::log(0.05F)}};
  std::vector<pcod_common::BoundingBox> boxes{a, b};
  pcod_common::NmsConfig config;
  config.merge_enabled = true;
  config.score_thresholds = {0.0F};
  pcod_common::ApplyRotatedNms(boxes, config);
  if (boxes.size() != 1 || boxes[0].classification.size() != 3) {
    throw std::runtime_error("Merging lost a class hypothesis");
  }
  std::vector<perception_msgs::msg::ObjectClassification> classes;
  for (const auto& entry : boxes[0].classification) {
    perception_msgs::msg::ObjectClassification value;
    value.type = static_cast<uint8_t>(entry.class_idx);
    value.probability = entry.score;
    classes.push_back(value);
  }
  point_cloud_object_detection::sanitize_classifications(classes, 0.0F, true);
  double total = 0.0;
  for (const auto& entry : classes) {
    const std::array<double, 3> expected{0.52, 0.43, 0.05};
    expectNear(entry.probability, expected.at(entry.type));
    total += entry.probability;
  }
  expectNear(total, 1.0);
  return 0;
}
