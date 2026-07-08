// Copyright Institute for Automotive Engineering (ika), RWTH Aachen University
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <perception_msgs/msg/object_classification.hpp>
#include <string>
#include <vector>

namespace point_cloud_object_detection {
/**
 * @brief Convert an arbitrary class name into a topic-safe suffix.
 */
std::string sanitizeTopicName(const std::string& class_name);

/**
 * @brief Own implementation of std::clamp, since this is only available
 * starting from C++17
 *
 */
template <class T>
constexpr const T& clamp(const T& value, const T& low, const T& high) {
  assert(!(high < low));
  if (value < low) {
    return low;
  }
  if (high < value) {
    return high;
  }
  return value;
}

/**
 * @brief This method cleans up the classifications that are outputted by the
 * model.
 *
 * First, it performs a softmax on the scores, if needed (for PBOD).
 * Then, it removes all classes with a score below the threshold.
 * Finally, the remaining classes are sorted and their scores normalized.
 * If no class is above the threshold, the UNKNOWN class is assigned.
 *
 * @param classifications Original classifications together with their scores,
 * will be modified in place
 * @param threshold The threshold to use for removing classes
 * @param softmax Flag to indicate if a softmax should be performed on the
 * scores, default is false.
 */
void sanitize_classifications(std::vector<perception_msgs::msg::ObjectClassification>& classifications,
                              float threshold,
                              bool softmax = false);

/**
 * @brief Computes log(exp(x)+exp(y)) in a numerically stable way.
 *
 * @tparam T floating point type
 * @param x summand 1 as logit
 * @param y summand 2 as logit
 * @return T sum of values as logit
 */
template <typename T>
inline static T logaddexp(T x, T y) {
  if (x == y) {
    return x + static_cast<T>(M_LN2);
  }
  const T tmp = x - y;

  if (tmp > 0) {
    return x + std::log1p(std::exp(-tmp));
  } else {
    return y + std::log1p(std::exp(tmp));
  }
}

}  // namespace point_cloud_object_detection
