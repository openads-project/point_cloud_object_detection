#include "point_cloud_object_detection/NonMaxSuppression.hpp"
#include "point_cloud_object_detection/Utils.hpp"

namespace point_cloud_object_detection {

const float NonMaxSuppression::kInternalScoreThreshold = 0.5;

NonMaxSuppression::NonMaxSuppression(ModelConfig& model_config, Params& params)
    : model_config_{model_config}, params_{params} {}

void NonMaxSuppression::nms(std::vector<BoundingBox>& bboxes) {
  // sort by existence probability
  std::vector<std::pair<float, BoundingBox*>> bboxes_with_internal_scores;
  bboxes_with_internal_scores.reserve(bboxes.size());
  std::transform(bboxes.begin(), bboxes.end(), std::back_inserter(bboxes_with_internal_scores),
                 [this](BoundingBox& bbox) {
                   return std::make_pair(
                       scale_score(bbox.existence_probability,
                                   static_cast<float>(model_config_.nms_score_threshold.at(
                                       std::max_element(bbox.classification.begin(), bbox.classification.end(),
                                                        [](const ClassificationEntry& a, const ClassificationEntry& b) {
                                                          return a.score < b.score;
                                                        })
                                           ->class_idx)),
                                   kInternalScoreThreshold),
                       &bbox);
                 });
  std::sort(bboxes_with_internal_scores.begin(), bboxes_with_internal_scores.end(),
            [](const std::pair<float, BoundingBox*>& a, const std::pair<float, BoundingBox*>& b) {
              return a.first > b.first;
            });

  std::vector<BoundingBox> nms_bboxes;
  for (auto& bbox : bboxes_with_internal_scores) {
    if (bbox.first < kInternalScoreThreshold) {
      break;
    }
    bool keep = true;
    for (const auto& nms_bbox : nms_bboxes) {
      if (bbox.second->overlaps(nms_bbox, model_config_.nms_iou_threshold)) {
        keep = false;
        break;
      }
    }
    if (keep) {
      // Normalize classification, and for PBOD, also the score
      nms_bboxes.push_back(*bbox.second);
      if (nms_bboxes.size() >= static_cast<std::size_t>(model_config_.nms_max_num_objects)) {
        break;
      }
    }
  }
  bboxes = nms_bboxes;
}

float BoundingBox::intersection_area(const BoundingBox& other) const {
  // C++ version of https://stackoverflow.com/a/45268241
  auto rect1 = rectangle_vertices();
  auto rect2 = other.rectangle_vertices();

  std::vector<BoundingBoxVertex> intersection = rect1;

  for (std::size_t i = 0; i < rect2.size(); ++i) {
    if (intersection.size() <= 2) {
      break;  // No intersection
    }

    Line line(rect2[i], rect2[(i + 1) % rect2.size()]);
    std::vector<BoundingBoxVertex> new_intersection;
    std::vector<float> line_values;

    for (const auto& t : intersection) {
      line_values.push_back(line(t));
    }

    for (std::size_t j = 0; j < intersection.size(); ++j) {
      const BoundingBoxVertex& s = intersection[j];
      const BoundingBoxVertex& t = intersection[(j + 1) % intersection.size()];
      float s_value = line_values[j];
      float t_value = line_values[(j + 1) % line_values.size()];

      if (s_value <= 0) {
        new_intersection.push_back(s);
      }

      if (s_value * t_value < 0) {
        BoundingBoxVertex intersection_point = line.intersection(Line(s, t));
        new_intersection.push_back(intersection_point);
      }
    }

    intersection = new_intersection;
  }

  if (intersection.size() <= 2) {
    return 0.0;
  }

  float area = 0.0;
  for (std::size_t i = 0; i < intersection.size(); ++i) {
    const BoundingBoxVertex& p = intersection[i];
    const BoundingBoxVertex& q = intersection[(i + 1) % intersection.size()];
    area += p.cross(q);
  }

  return 0.5 * std::abs(area);
}

bool BoundingBox::overlaps(const BoundingBox& other, float iou_threshold) const {
  float intersection = intersection_area(other);
  float union_area = length * width + other.length * other.width - intersection;
  float iou = intersection / union_area;
  return iou > iou_threshold;
}

// Function to calculate rectangle vertices
std::vector<BoundingBoxVertex> BoundingBox::rectangle_vertices() const {
  float angle = yaw;
  float dx = length / 2.0;
  float dy = width / 2.0;
  float dxcos = dx * cos(angle);
  float dxsin = dx * sin(angle);
  float dycos = dy * cos(angle);
  float dysin = dy * sin(angle);

  std::vector<BoundingBoxVertex> vertices;
  vertices.emplace_back(center[0] + (-dxcos - -dysin), center[1] + (-dxsin + -dycos));
  vertices.emplace_back(center[0] + (dxcos - -dysin), center[1] + (dxsin + -dycos));
  vertices.emplace_back(center[0] + (dxcos - dysin), center[1] + (dxsin + dycos));
  vertices.emplace_back(center[0] + (-dxcos - dysin), center[1] + (-dxsin + dycos));
  return vertices;
}

}  // namespace point_cloud_object_detection
