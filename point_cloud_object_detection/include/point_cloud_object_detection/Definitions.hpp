#pragma once

#include <tf2/transform_datatypes.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include <Eigen/Dense>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2_sensor_msgs/tf2_sensor_msgs.hpp>

namespace point_cloud_object_detection {
struct Anchor {
  float height;
  float width;
  float length;
  float z_center;
  float yaw;
};

/**
 * @brief One entry in a vector of possible classifications
 * 
 */
struct ClassificationEntry {
  std::size_t class_idx;
  float score;  // We don't call it probability, as it is not necessarily normalized
};

struct BoundingBoxVertex {
  // Coordinates of the vertex
  float x, y;

  BoundingBoxVertex(float x = 0, float y = 0) : x(x), y(y) {}

  BoundingBoxVertex operator+(const BoundingBoxVertex& v) const { return BoundingBoxVertex(x + v.x, y + v.y); }

  BoundingBoxVertex operator-(const BoundingBoxVertex& v) const { return BoundingBoxVertex(x - v.x, y - v.y); }

  // Cross product of the vectors to two vertices, aka signed area of the parallelogram spanned by the two vectors
  float cross(const BoundingBoxVertex& v) const { return x * v.y - y * v.x; }
};

/**
 * @brief A class representing a line in 2D space as ax + by + c = 0
 * 
 * All methods are defined in the class definition, so they are inlined.
 */
struct Line {
  // Coefficients of the line
  float a, b, c;

  /**
   * @brief Constructs a Line linking two vertices
   * 
   * @param v1 Vertex 1
   * @param v2 Vertex 2
   */
  Line(const BoundingBoxVertex& v1, const BoundingBoxVertex& v2) {
    a = v2.y - v1.y;
    b = v1.x - v2.x;
    c = v2.cross(v1);
  }

  /**
   * @brief Computes the signed and scaled distance of a point to the line
   * 
   * @param p The point
   * @return float The signed and scaled distance
   */
  float operator()(const BoundingBoxVertex& p) const { return a * p.x + b * p.y + c; }

  /**
   * @brief Computes the point where two lines intersect
   * 
   * @param other 
   * @return BoundingBoxVertex 
   */
  BoundingBoxVertex intersection(const Line& other) const {
    float w = a * other.b - b * other.a;
    return BoundingBoxVertex((b * other.c - c * other.b) / w, (c * other.a - a * other.c) / w);
  }
};

/**
 * @brief A BoundingBox, or Object, in 3D space with classification and existence probability
 * 
 */
struct BoundingBox {
  // Cartesian coordinates to describe the bounding box center in the x, y plane
  std::array<float, 2> center;
  float z;
  float length;
  float width;
  float height;
  float yaw;
  float existence_probability;
  // As the classification is usually uncertain, multiple hypotheses can be stored here wih their respective probability.
  std::vector<ClassificationEntry> classification;
  // Some models also predict a velocity, other don't.
  bool has_velocity = false;
  float v_x = 0.;
  float v_y = 0.;

  /**
   * @brief Calculates wether this BoundinfBox overlaps with another one with an IoU of at least the given threshold
   * 
   * @param other second BoundingBox
   * @param iou_threshold threshold for the IoU
   * @return IoU > iou_threshold 
   */
  bool overlaps(const BoundingBox& other, float iou_threshold) const;

  /**
   * @brief Computes the intersection area of two BoundingBoxes
   * 
   * @param other second BoundingBox
   * @return the Area of the intersection, or 0 if there is no intersection
   */
  float intersection_area(const BoundingBox& other) const;

  /**
   * @brief Generates the vertices of the BoundingBox in 2D
   * 
   * @return std::vector<BoundingBoxVertex> vector of vertices
   */
  std::vector<BoundingBoxVertex> rectangle_vertices() const;
};

// model config
struct ModelConfig {
  // Preprocessing
  int intensity_threshold;

  // Postprocessing
  std::vector<std::string> predicted_class_names;
  std::map<std::string, uint8_t> class_mapping_;
  bool with_velocity;

  // Grid
  float x_min;
  float x_max;
  float y_min;
  float y_max;
  float z_min;
  float z_max;
  int x_grid_size;
  int y_grid_size;

  // NMS
  int nms_max_num_objects;
  float nms_iou_threshold;
  std::vector<double> nms_score_threshold;

  // PP specific config
  int max_pillars;
  int max_points_per_pillar;
  int n_features;
  int downscaling;
  float delta_x;
  float delta_y;
  std::string anchors_string;
  std::vector<Anchor> anchor_boxes;
  std::vector<double> anchor_diagonals;
  float min_nms_score_threshold;

  // PBOD specific config, also needed for TPOD as PBOD is its backbone
  int max_num_points;
  int num_point_features = 1;
  std::string point_type = "PointXYZI";
  std::vector<int64_t> stride;
  int first_up_stride;

  // TPOD specific config
  float cls_threshold;

  std::vector<int64_t> pillar_map_size;
  std::vector<std::vector<float>> pillar_map_range;

  // Legacy compatibility
  bool mask_is_bool;
  bool zero_intensity;

  // No-detection zone point filtering (in inference_frame)
  bool no_detection_zone_remove_points = false;  // If true, drop raw points in the zone from model input
  double no_detection_zone_x_min = 0.0;
  double no_detection_zone_x_max = 0.0;
  double no_detection_zone_y_min = 0.0;
  double no_detection_zone_y_max = 0.0;

  // Detection area (circular sector) point filtering (in inference_frame)
  // If true, drop raw points outside the sector from model input
  bool detection_area_remove_points_outside = false;
  double detection_area_center_x = 0.0;     // sector center x (m)
  double detection_area_center_y = 0.0;     // sector center y (m)
  double detection_area_radius = 0.0;       // sector radius (m)
  double detection_area_bearing_deg = 0.0;  // sector central azimuth (deg, 0 along +x, CCW positive)
  double detection_area_fov_deg = 360.0;    // sector FOV angle (deg)
};

// parameters
struct Params {
  std::string model_name;
  std::string model_version;
  std::string server_url;
  bool use_shm;

  std::string inference_frame;
  std::string output_frame;

  uint64_t sensor_id;

  std::vector<double> variance;
  double class_score_threshold;

  // Optional no-detection rectangle (in inference_frame) where detections are not allowed
  bool no_detection_zone_enabled = false;
  double no_detection_zone_x_min = 0.0;
  double no_detection_zone_x_max = 0.0;
  double no_detection_zone_y_min = 0.0;
  double no_detection_zone_y_max = 0.0;
  // If true, publish the bounds of the no-detection zone as a PolygonStamped
  bool no_detection_zone_publish_polygon = false;
  // If true, publish raw points that lie inside the no-detection zone
  bool no_detection_zone_publish_points = false;

  // Detection area (circular sector) in inference_frame where detections are allowed
  bool detection_area_enabled = false;  // turns on sector-based filtering/publishing
  double detection_area_center_x = 0.0;
  double detection_area_center_y = 0.0;
  double detection_area_radius = 0.0;
  double detection_area_bearing_deg = 0.0;  // central azimuth (deg, 0 along +x, CCW)
  double detection_area_fov_deg = 360.0;    // angular width (deg)
  bool detection_area_publish_polygon = false;
  int detection_area_num_segments = 32;  // polygon approximation of arc

  // Detection filtering mode
  // If enabled, remove detections that lie outside the area either by center or completely
  bool detection_area_filter_detections = false;      // enable detection filtering by sector
  std::string detection_area_filter_mode = "center";  // "center" or "complete"

  // Model bounds polygon publication (XY rectangle from x_min/x_max/y_min/y_max)
  bool model_bounds_publish_polygon = false;
};

}  // namespace point_cloud_object_detection
