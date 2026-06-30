// Copyright Institute for Automotive Engineering (ika), RWTH Aachen University
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace point_cloud_object_detection {

using Point = pcl::PointXYZI;
using PointCloud = pcl::PointCloud<Point>;

}  // namespace point_cloud_object_detection
