// Copyright Institute for Automotive Engineering (ika), RWTH Aachen University
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <cmath>
#include <limits>
#include <optional>
#include <stdexcept>

#include <perception_msgs_utils/object_access.hpp>
#include <tf2_perception_msgs/tf2_perception_msgs.hpp>

#include "pcod_common/nms.hpp"
#include "pcod_common/pbod_postprocess.hpp"
#include "point_cloud_object_detection/Utils.hpp"

namespace {
void expectNear(double actual, double expected) {
  if (!std::isfinite(actual) || std::abs(actual - expected) > 1e-6) {
    throw std::runtime_error("Published detection confidence differs from selected score");
  }
}
}  // namespace

/** Verify score publication, transform preservation, and rejection of invalid scores. */
int main() {
  // Distinct input probabilities give each score mode a different expected result.
  const std::array<float, 1> objectness = {std::log(9.0F)};     // presence = 0.9
  const std::array<float, 1> focal = {0.0F};                    // presence * quality = 0.5
  const std::array<float, 2> classes = {std::log(3.0F), 0.0F};  // best class = 0.75
  const std::array<float, 6> sizes = {1, 1, 1, 1, 1, 1};
  const std::array<float, 14> regression = {};
  pcod_common::PbodOutputsView view;
  view.objectness_logits = objectness.data();
  view.focal_logits = focal.data();
  view.class_logits = classes.data();
  view.size_posterior = sizes.data();
  view.reg_logits = regression.data();
  view.num_pillars = 1;
  view.num_classes = 2;
  view.reg_dim = 7;
  const auto grid = pcod_common::BuildPillarGrid({1, 1}, {{{0, 1}, {0, 1}, {0, 1}}}, 1, 1);
  const std::array<std::pair<pcod_common::PbodScoreMode, double>, 3> cases = {
      {{pcod_common::PbodScoreMode::Existence, 0.9},
       {pcod_common::PbodScoreMode::ExistenceQuality, 0.5},
       {pcod_common::PbodScoreMode::ExistenceQualityClass, 0.375}}};
  for (const auto& [mode, expected] : cases) {
    pcod_common::PbodPostprocessConfig config;
    config.score_mode = mode;
    auto boxes = pcod_common::DecodePbod(view, grid, config);
    if (boxes.size() != 1) throw std::runtime_error("Expected one decoded box");
    // Check merging retains the selected representative score.
    auto weaker = boxes.front();
    weaker.detection_score = static_cast<float>(expected / 2);
    boxes.push_back(weaker);
    pcod_common::NmsConfig nms;
    nms.merge_enabled = true;
    nms.score_thresholds = {0.0F};
    pcod_common::ApplyRotatedNms(boxes, nms);
    if (boxes.size() != 1) throw std::runtime_error("Expected merged representative");
    expectNear(boxes.front().existence_probability, 0.9);
    perception_msgs::msg::Object object;
    perception_msgs::object_access::initializeState(object, perception_msgs::msg::ISCACTR::MODEL_ID);
    perception_msgs::object_access::setContinuousStateCovarianceDiagonal(object, std::vector<double>(12, 1.0));
    point_cloud_object_detection::setDetectionConfidence(boxes.front(), object);
    expectNear(object.existence_probability, expected);

    perception_msgs::msg::ObjectList list, transformed;
    list.header.frame_id = "sensor";
    list.objects.push_back(object);
    geometry_msgs::msg::TransformStamped transform;
    transform.header.frame_id = "map";
    transform.child_frame_id = "sensor";
    transform.transform.rotation.w = 1.0;
    transform.transform.translation.x = 10.0;
    tf2::doTransform(list, transformed, transform);
    if (transformed.objects.size() != 1) throw std::runtime_error("Transform lost object");
    expectNear(transformed.objects.front().existence_probability, expected);
  }

  // Reject invalid detection scores even when the presence probability is valid.
  const std::array<std::optional<float>, 5> invalid = {std::nullopt, std::numeric_limits<float>::quiet_NaN(),
                                                       std::numeric_limits<float>::infinity(), -0.1F, 1.1F};
  for (const auto score : invalid) {
    pcod_common::BoundingBox box;
    box.existence_probability = 0.9F;
    box.detection_score = score;
    perception_msgs::msg::Object object;
    bool rejected = false;
    try {
      point_cloud_object_detection::setDetectionConfidence(box, object);
    } catch (const std::invalid_argument&) {
      rejected = true;
    }
    if (!rejected) throw std::runtime_error("Invalid detection score was accepted");
  }
  for (float score : {0.0F, 1.0F}) {
    pcod_common::BoundingBox box;
    box.detection_score = score;
    perception_msgs::msg::Object object;
    point_cloud_object_detection::setDetectionConfidence(box, object);
    expectNear(object.existence_probability, score);
  }
  return 0;
}
