#pragma once

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <Eigen/Core>

namespace point_cloud_object_detection {

struct PointXYZRV {
  PCL_ADD_POINT4D;
  float reflectivity = 0.0f;
  float velocity = 0.0f;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

using Point = pcl::PointXYZI;
using PointCloud = pcl::PointCloud<Point>;

}  // namespace point_cloud_object_detection

POINT_CLOUD_REGISTER_POINT_STRUCT(point_cloud_object_detection::PointXYZRV,
                                  (float, x, x)(float, y, y)(float, z, z)(float, reflectivity,
                                                                          reflectivity)(float, velocity, velocity))
