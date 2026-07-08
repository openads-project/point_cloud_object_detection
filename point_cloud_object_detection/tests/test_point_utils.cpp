// Copyright Institute for Automotive Engineering (ika), RWTH Aachen University
// SPDX-License-Identifier: Apache-2.0

#include "point_cloud_object_detection/PointUtils.hpp"

#include <cmath>
#include <stdexcept>

namespace {

void expectNear(float actual, float expected) {
  if (std::abs(actual - expected) >= 1e-6F) {
    throw std::runtime_error("unexpected PCL point value");
  }
}

}  // namespace

int main() {
  const auto point = point_cloud_object_detection::makePoint(1.25F, -2.5F, 3.75F, 42.0F);

  expectNear(point_cloud_object_detection::getPointX(point), 1.25F);
  expectNear(point_cloud_object_detection::getPointY(point), -2.5F);
  expectNear(point_cloud_object_detection::getPointZ(point), 3.75F);
  expectNear(point_cloud_object_detection::getPointIntensity(point), 42.0F);

  return 0;
}
