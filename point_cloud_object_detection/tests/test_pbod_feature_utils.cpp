// Copyright Institute for Automotive Engineering (ika), RWTH Aachen University
// SPDX-License-Identifier: Apache-2.0

#include "point_cloud_object_detection/PBODFeatureUtils.hpp"

#include <cmath>
#include <stdexcept>

namespace {

void expectNear(float actual, float expected) {
  if (std::abs(actual - expected) >= 1e-6F) {
    throw std::runtime_error("unexpected PBOD feature row value");
  }
}

}  // namespace

int main() {
  using namespace point_cloud_object_detection;

  const auto row = makePbodFeatureRow({3.0F, 4.0F, -1.5F, 0.25F, 2.0F, 1.0F, -2.0F, 0.1F, 0.2F, 0.3F, 2.5F, 3.5F, 0.0F, -4.0F});

  static_assert(row.size() == kPbodPreprocessedFeatureDim);
  expectNear(row[kPbodFeatureX], 3.0F);
  expectNear(row[kPbodFeatureY], 4.0F);
  expectNear(row[kPbodFeatureZ], -1.5F);
  expectNear(row[kPbodFeatureRadius], 5.0F);
  expectNear(row[kPbodFeatureZRelative], 2.5F);
  expectNear(row[kPbodFeatureInverseRadius], 0.2F);
  expectNear(row[kPbodFeatureSinTheta], 0.8F);
  expectNear(row[kPbodFeatureCosTheta], 0.6F);
  expectNear(row[kPbodFeatureVarianceX], 0.1F);
  expectNear(row[kPbodFeatureVarianceY], 0.2F);
  expectNear(row[kPbodFeatureVarianceZ], 0.3F);
  expectNear(row[kPbodFeatureIntensity], 0.25F);
  expectNear(row[kPbodFeatureClusterOffsetX], 1.0F);
  expectNear(row[kPbodFeatureClusterOffsetY], 3.0F);
  expectNear(row[kPbodFeatureClusterOffsetZ], 0.5F);
  expectNear(row[kPbodFeatureCenterOffsetX], 0.5F);
  expectNear(row[kPbodFeatureCenterOffsetY], 0.5F);
  expectNear(row[kPbodFeatureCenterOffsetZ], -1.5F);

  const auto origin_row = makePbodFeatureRow({});
  expectNear(origin_row[kPbodFeatureRadius], 0.0F);
  expectNear(origin_row[kPbodFeatureInverseRadius], 1000000.0F);
  expectNear(origin_row[kPbodFeatureSinTheta], 0.0F);
  expectNear(origin_row[kPbodFeatureCosTheta], 1.0F);

  return 0;
}
