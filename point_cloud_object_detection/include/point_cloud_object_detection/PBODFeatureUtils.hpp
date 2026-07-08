// Copyright Institute for Automotive Engineering (ika), RWTH Aachen University
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cmath>

namespace point_cloud_object_detection {

/**
 * @brief Number of model input features per preprocessed PBOD point.
 */
constexpr int kPbodPreprocessedFeatureDim = 18;

/**
 * @brief Indices of the preprocessed PBOD feature row.
 */
enum PbodFeatureIndex : int {
  kPbodFeatureX = 0,
  kPbodFeatureY,
  kPbodFeatureZ,
  kPbodFeatureRadius,
  kPbodFeatureZRelative,
  kPbodFeatureInverseRadius,
  kPbodFeatureSinTheta,
  kPbodFeatureCosTheta,
  kPbodFeatureVarianceX,
  kPbodFeatureVarianceY,
  kPbodFeatureVarianceZ,
  kPbodFeatureIntensity,
  kPbodFeatureClusterOffsetX,
  kPbodFeatureClusterOffsetY,
  kPbodFeatureClusterOffsetZ,
  kPbodFeatureCenterOffsetX,
  kPbodFeatureCenterOffsetY,
  kPbodFeatureCenterOffsetZ
};

/**
 * @brief Inputs required to compute one preprocessed PBOD feature row.
 */
struct PbodFeatureRowInputs {
  float x = 0.0F;
  float y = 0.0F;
  float z = 0.0F;
  float intensity = 0.0F;
  float mean_x = 0.0F;
  float mean_y = 0.0F;
  float mean_z = 0.0F;
  float variance_x = 0.0F;
  float variance_y = 0.0F;
  float variance_z = 0.0F;
  float center_x = 0.0F;
  float center_y = 0.0F;
  float center_z = 0.0F;
  float z_min = 0.0F;
};

/**
 * @brief Build one PBOD feature row from raw point, cluster, and grid-center values.
 */
inline std::array<float, kPbodPreprocessedFeatureDim> makePbodFeatureRow(const PbodFeatureRowInputs& inputs) {
  constexpr float kRadiusEpsilon = 1e-6F;
  const float cluster_x = inputs.x - inputs.mean_x;
  const float cluster_y = inputs.y - inputs.mean_y;
  const float cluster_z = inputs.z - inputs.mean_z;
  const float center_offset_x = inputs.x - inputs.center_x;
  const float center_offset_y = inputs.y - inputs.center_y;
  const float center_offset_z = inputs.z - inputs.center_z;
  const float radius_squared = inputs.x * inputs.x + inputs.y * inputs.y;
  const float radius = std::sqrt(radius_squared);
  const float inverse_radius = 1.0F / (radius > kRadiusEpsilon ? radius : kRadiusEpsilon);
  float sin_theta = 0.0F;
  float cos_theta = 1.0F;
  if (radius > 0.0F) {
    const float inverse_norm = 1.0F / radius;
    sin_theta = inputs.y * inverse_norm;
    cos_theta = inputs.x * inverse_norm;
  }

  return {inputs.x,
          inputs.y,
          inputs.z,
          radius,
          inputs.z - inputs.z_min,
          inverse_radius,
          sin_theta,
          cos_theta,
          inputs.variance_x,
          inputs.variance_y,
          inputs.variance_z,
          inputs.intensity,
          cluster_x,
          cluster_y,
          cluster_z,
          center_offset_x,
          center_offset_y,
          center_offset_z};
}

}  // namespace point_cloud_object_detection
