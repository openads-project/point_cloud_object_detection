#include "point_cloud_object_detection/Utils.hpp"

namespace point_cloud_object_detection {
std::vector<float> linspace(float start_in, float end_in, int num_in) {
  std::vector<float> linspaced;

  if (num_in == 0) {
    return linspaced;
  }
  if (num_in == 1) {
    linspaced.push_back(start_in);
    return linspaced;
  }

  float delta = (end_in - start_in) / (static_cast<float>(num_in) - 1);

  for (int i = 0; i < num_in - 1; ++i) {
    linspaced.push_back(start_in + delta * static_cast<float>(i));
  }
  linspaced.push_back(end_in);  // I want to ensure that start and end
                                // are exactly the same as the input
  return linspaced;
}

// Sanitizes a ROS topic name by removing leading special characters and replacing invalid characters with underscores.
std::string sanitizeTopicName(const std::string& name) {
  std::string sanitized = name;

  // Remove all leading '~' or '/' characters.
  auto first_valid = sanitized.find_first_not_of("~/");
  sanitized = (first_valid == std::string::npos) ? "" : sanitized.substr(first_valid);

  // Replace invalid characters with underscores.
  for (char& c : sanitized) {
    if (!std::isalnum(static_cast<unsigned char>(c)) && c != '_' && c != '~' && c != '{' && c != '}') {
      c = '_';
    }
  }

  return sanitized;
}

float wrap_to_range(float val, float min_val, float max_val) {
  float range = max_val - min_val;
  return val - range * std::floor((val - min_val) / range);
}

std::size_t IntPairHash::operator()(const std::pair<uint32_t, uint32_t>& pair) const {
  // assert(sizeof(std::size_t) >= 8);
  // shift first integer over to make room for the second integer.
  // The two are then packed side by side.
  const uint64_t shift = 32;
  return (static_cast<uint64_t>(pair.first) << shift) | (static_cast<uint64_t>(pair.second));
}

void sanitize_classifications(std::vector<perception_msgs::msg::ObjectClassification>& classifications, float threshold,
                              bool softmax) {
  // Compute softmax from logits (needed for PBOD)
  if (softmax) {
    double total_score = 0;
    for (auto& classification : classifications) {
      classification.probability = std::exp(classification.probability);
      total_score += classification.probability;
    }
    for (auto& classification : classifications) {
      classification.probability /= total_score;
    }
  }

  // Remove classifications with a score lower than the threshold and count the sum of remaining scores
  double total_score = 0;
  for (auto classification = classifications.begin(); classification != classifications.end();) {
    if (classification->probability < threshold) {
      classification = classifications.erase(classification);
    } else {
      total_score += classification->probability;
      ++classification;
    }
  }

  if (classifications.empty()) {
    // If no type has a score of more than the threshold, use unknwon (which is actually made for exactly this case)
    perception_msgs::msg::ObjectClassification classification;
    classification.probability = 1;
    classification.type = perception_msgs::msg::ObjectClassification::UNKNOWN;
    classifications.push_back(classification);

  } else {
    // Sort descending and normalize
    std::sort(classifications.begin(), classifications.end(),
              [](const perception_msgs::msg::ObjectClassification& first,
                 const perception_msgs::msg::ObjectClassification& second) {
                return first.probability > second.probability;
              });
    for (auto& classification : classifications) {
      classification.probability /= total_score;
    }
  }
}

}  // namespace point_cloud_object_detection
