// Copyright Institute for Automotive Engineering (ika), RWTH Aachen University
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pcl/point_struct_traits.h>
#include <pcl/type_traits.h>

#include "point_cloud_object_detection/PointTypes.hpp"

namespace point_cloud_object_detection {

inline float getPointX(const Point& point) {
  float value = 0.0F;
  pcl::getFieldValue(point, pcl::traits::offset<Point, pcl::fields::x>::value, value);
  return value;
}

inline float getPointY(const Point& point) {
  float value = 0.0F;
  pcl::getFieldValue(point, pcl::traits::offset<Point, pcl::fields::y>::value, value);
  return value;
}

inline float getPointZ(const Point& point) {
  float value = 0.0F;
  pcl::getFieldValue(point, pcl::traits::offset<Point, pcl::fields::z>::value, value);
  return value;
}

inline float getPointIntensity(const Point& point) {
  float value = 0.0F;
  pcl::getFieldValue(point, pcl::traits::offset<Point, pcl::fields::intensity>::value, value);
  return value;
}

inline void setPointFloatField(Point& point, std::size_t field_offset, float field_value) {
  pcl::setFieldValue(point, field_offset, field_value);
}

inline Point makePoint(float x, float y, float z, float intensity) {
  Point point;
  constexpr std::size_t x_offset = pcl::traits::offset<Point, pcl::fields::x>::value;
  constexpr std::size_t y_offset = pcl::traits::offset<Point, pcl::fields::y>::value;
  constexpr std::size_t z_offset = pcl::traits::offset<Point, pcl::fields::z>::value;
  constexpr std::size_t intensity_offset = pcl::traits::offset<Point, pcl::fields::intensity>::value;
  setPointFloatField(point, x_offset, x);
  setPointFloatField(point, y_offset, y);
  setPointFloatField(point, z_offset, z);
  setPointFloatField(point, intensity_offset, intensity);
  return point;
}

}  // namespace point_cloud_object_detection
