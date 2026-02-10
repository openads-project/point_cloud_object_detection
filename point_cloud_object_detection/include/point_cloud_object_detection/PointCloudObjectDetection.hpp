#pragma once

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <tf2/exceptions.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <triton_cpp/triton_interface.hpp>

#include <chrono>
#include <cstddef>
#include <perception_msgs/msg/object.hpp>
#include <perception_msgs/msg/object_list.hpp>
#include <perception_msgs_utils/object_access.hpp>
#include <point_cloud_transport/point_cloud_transport.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>
#include <string>
#include <tf2_perception_msgs/tf2_perception_msgs.hpp>
#include <tf2_sensor_msgs/tf2_sensor_msgs.hpp>
#include <vector>

#include "pcod_common/nms.hpp"
#include "point_cloud_object_detection/Definitions.hpp"
#include "point_cloud_object_detection/Model.hpp"
#include "point_cloud_object_detection/PBODModel.hpp"
#include "point_cloud_object_detection/PointTypes.hpp"

#include <geometry_msgs/msg/polygon_stamped.hpp>

namespace point_cloud_object_detection {
using namespace std::chrono_literals;

// namespace acronyms
namespace pm = perception_msgs;

class PointCloudObjectDetection : public rclcpp::Node {
 public:
  /**
  * @brief Constructor getting its options e.g. from ComposableNodeContainer
  * @param options NodeOptions
  */
  explicit PointCloudObjectDetection(const rclcpp::NodeOptions& options);

 protected:
  /**
   * @brief Declares all parameters that this node uses no matter which architecture is used
   */
  void declareParameters();
  /**
   * @brief Loads all ROS parameters for the node itself
   */
  void loadParameters();
  /**
   * @brief Loads all ROS parameters for the model depending on the architecture
   */
  void loadModelConfig();

  /**
   * @brief Declares a parameter, if it is not already declared
   * @tparam T anything convertible to either rclcpp::ParameterValue or rclcpp::ParameterType
   */
  template <typename T>
  void declare_parameter_if_not_exists(const std::string& name, const T& type_or_default, const std::string& desc);

  /**
   * @brief Callback for configurable parameters: Is executed every time a ROS parameter is modified
   *
   * @param parameters                                    Vector with all ROS parameters
   * @return rcl_interfaces::msg::SetParametersResult     Result of parameter modification
   */
  rcl_interfaces::msg::SetParametersResult parametersCallback(const std::vector<rclcpp::Parameter>& parameters);

  /**
   * @brief Tries to update the score thresholds for non-maximum suppression
   *
   * @param score_thresholds
   * @return true if the update was successful, i.e. the vector has the same size as the number of classes or 1
   */
  bool updateNMSScoreThreshold(std::vector<double>& score_thresholds);

  /**
   * @brief Setup of model, parameter callback and publisher/subscriber
   *
   */
  void setup();

  /**
   * @brief Initialize the Triton interface and detection model with new model name/version
   *
   */
  void initializeModel();

  /**
   * @brief Setup of publishers
   *
   */
  void setupPublishers();

  /**
   * @brief Transformation of point cloud coordinates into specified inference frame and transformation into pcl data type
   *
   * @param msg               Point cloud data in ROS message type format
   * @param point_cloud       Point cloud in pcl format -> Return reference
   */
  void processPointCloud(const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg, PointCloud& point_cloud);
  /**
   * @brief Create object list message type format using bounding box data
   *
   * @param bboxes            Vector containing all bounding boxes
   * @param object_list       Object list -> Return reference
   */
  void boxesToObjectList(const std::vector<BoundingBox>& bboxes, perception_msgs::msg::ObjectList& object_list);

  /**
   * @brief Callback executing the prediction every time a point cloud message is received by the ROS node
   *
   * @param msg       ROS point cloud message
   */
  void predict(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& pcl_msg);

  void validateParamsOrThrow() const;
  void validateModelConfigOrThrow() const;

  // constants
  static const std::string kInputTopic;
  static const std::string kOutputTopic;
  static const std::string kNoDetectionZoneTopic;
  static const std::string kNoDetectionZonePointsTopic;
  static const std::string kDetectionAreaTopic;
  static const std::string kModelBoundsTopic;
  static const std::map<uint8_t, std::vector<std::string>> kPossibleClassNames;

  // other member variables
  rclcpp::TimerBase::SharedPtr setup_timer_;
  std::unique_ptr<triton_cpp::TritonInterface> triton_interface_;

  // dynamic parameter callback
  OnSetParametersCallbackHandle::SharedPtr parameters_callback_;

  // publisher and subscriber
  std::shared_ptr<point_cloud_transport::Subscriber> subscriber_;
  rclcpp::Publisher<perception_msgs::msg::ObjectList>::SharedPtr publisher_;
  rclcpp::Publisher<geometry_msgs::msg::PolygonStamped>::SharedPtr no_detection_zone_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PolygonStamped>::SharedPtr detection_area_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PolygonStamped>::SharedPtr model_bounds_pub_;

  // publisher for raw points inside the no-detection zone
  std::shared_ptr<point_cloud_transport::Publisher> no_detection_zone_points_publisher_;

  // transform listener and transform buffer
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;

 private:
  Params params_;
  ModelConfig model_config_;

  std::unique_ptr<Model> detection_model_;
  pcod_common::NmsConfig nms_config_;
};

}  // namespace point_cloud_object_detection
