// Copyright Institute for Automotive Engineering (ika), RWTH Aachen University
// SPDX-License-Identifier: Apache-2.0

#pragma once

#define _USE_MATH_DEFINES
#include <Eigen/Dense>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <perception_msgs/msg/object_classification.hpp>
#include <unsupported/Eigen/CXX11/Tensor>
#include <vector>

namespace point_cloud_object_detection {
std::string sanitizeTopicName(const std::string& class_name);

std::vector<float> linspace(float start_in, float end_in, int num_in);

float wrap_to_range(float val, float min_val = -M_PI, float max_val = M_PI);

struct IntPairHash {
  std::size_t operator()(const std::pair<uint32_t, uint32_t>& pair) const;
};

/**
 * @brief Struct defining all dimensions of the input feature network of PointPillars
 *
 */
struct PillarPoint {
  float x = 0.0;
  float y = 0.0;
  float z = 0.0;
  float intensity = 0.0;
  float xc = 0.0;
  float yc = 0.0;
  float zc = 0.0;
};

/**
 * @brief Own implementation of std::clamp, since this is only available starting from C++17
 *
 */
template <class T>
constexpr const inline T& clamp(const T& value, const T& low, const T& high) {
  assert(!(high < low));
  return (value < low) ? low : (high < value) ? high : value;
}

/**
 * @brief This method cleans up the classifications that are outputted by the model.
 * 
 * First, it performs a softmax on the scores, if needed (for PBOD).
 * Then, it removes all classes with a score below the threshold.
 * Finally, the remaining classes are sorted and their scores normalized.
 * If no class is above the threshold, the UNKNOWN class is assigned.
 * 
 * @param classifications Original classifications together with their scores, will be modified in place
 * @param threshold The threshold to use for removing classes
 * @param softmax Flag to indicate if a softmax should be performed on the scores, default is false.
 */
void sanitize_classifications(std::vector<perception_msgs::msg::ObjectClassification>& classifications, float threshold,
                              bool softmax = false);

/**
 * @brief This method scales a score from one threshold to another.
 * It can be used e.g. before NMS, if the network outputs objects with an expected score threshold, but the NMS uses a different one.
 * Scores below old_thresh are scaled linearly, scores above old_thresh are scaled linearly as well, but with respect to 1-<score/old/new threshold>.
 * 
 * @tparam T 
 * @param score Old score
 * @param old_thresh Old threshold
 * @param new_thresh New threshold
 * @return T New score
 */
template <typename T>
T scale_score(T score, T old_thresh, T new_thresh) {
  const T ONE = static_cast<T>(1);
  if (score <= old_thresh) {
    return score * new_thresh / old_thresh;
  } else {
    return ONE - (ONE - score) * (ONE - new_thresh) / (ONE - old_thresh);
  }
}

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
