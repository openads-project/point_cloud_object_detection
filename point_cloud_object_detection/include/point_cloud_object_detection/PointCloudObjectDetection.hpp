#pragma once

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <tf2/exceptions.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <triton_cpp/triton_interface.hpp>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <optional>
#include <perception_msgs/msg/object.hpp>
#include <perception_msgs/msg/object_list.hpp>
#include <perception_msgs_utils/object_access.hpp>
#include <point_cloud_transport/point_cloud_transport.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>
#include <string>
#include <tf2_perception_msgs/tf2_perception_msgs.hpp>
#include <tf2_sensor_msgs/tf2_sensor_msgs.hpp>
#include <tuple>
#include <type_traits>
#include <vector>

#include "pcod_common/nms.hpp"
#include "point_cloud_object_detection/Definitions.hpp"
#include "point_cloud_object_detection/Model.hpp"
#include "point_cloud_object_detection/PBODModel.hpp"
#include "point_cloud_object_detection/PointTypes.hpp"

#include <geometry_msgs/msg/polygon_stamped.hpp>

namespace point_cloud_object_detection {
using namespace std::chrono_literals;

template <typename C>
struct is_vector : std::false_type {};
template <typename T, typename A>
struct is_vector<std::vector<T, A>> : std::true_type {};
template <typename C>
inline constexpr bool is_vector_v = is_vector<C>::value;

// namespace acronyms
namespace pm = perception_msgs;

class PointCloudObjectDetection : public rclcpp::Node {
 public:
  /**
  * @brief Constructor getting its options e.g. from ComposableNodeContainer
  * @param options NodeOptions
  */
  explicit PointCloudObjectDetection(const rclcpp::NodeOptions& options);
  ~PointCloudObjectDetection() override;

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
   * @brief Synchronize model runtime options from the loaded node parameters
   */
  void syncModelRuntimeConfigFromParams();
  void syncModelRuntimeConfigFromParams(ModelConfig& model_config, const Params& params) const;
  /**
   * @brief Apply optional NMS overrides from node params onto manifest-derived model config
   */
  void syncNmsRuntimeConfigFromParams();
  void syncNmsRuntimeConfigFromParams(ModelConfig& model_config, pcod_common::NmsConfig& nms_config,
                                      const Params& params) const;
  /**
   * @brief Loads all ROS parameters for the model depending on the architecture
   */
  ModelConfig loadModelConfig(const Params& params, std::string& model_name, std::string& model_version,
                              pcod_common::NmsConfig& nms_config) const;

  template <typename T>
  void declareAndLoadParameter(const std::string& name, T& param, const std::string& description,
                               const bool add_to_auto_reconfigurable_params = true, const bool is_required = false,
                               const bool read_only = false, const std::optional<double>& from_value = std::nullopt,
                               const std::optional<double>& to_value = std::nullopt,
                               const std::optional<double>& step_value = std::nullopt,
                               const std::string& additional_constraints = "");

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
  bool updateNMSScoreThreshold(std::vector<double>& score_thresholds,
                               const std::vector<std::string>& predicted_class_names) const;

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
  void refreshResolvedModelConfigLocked();

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
  void processPointCloud(const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg, const ModelConfig& model_config,
                         const Params& params, PointCloud& point_cloud);
  /**
   * @brief Create object list message type format using bounding box data
   *
   * @param bboxes            Vector containing all bounding boxes
   * @param object_list       Object list -> Return reference
   */
  void boxesToObjectList(const std::vector<BoundingBox>& bboxes, const ModelConfig& model_config, const Params& params,
                         perception_msgs::msg::ObjectList& object_list);

  /**
   * @brief Callback executing the prediction every time a point cloud message is received by the ROS node
   *
   * @param msg       ROS point cloud message
   */
  void predict(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& pcl_msg);

  void validateParamsOrThrow() const;
  void validateModelConfigOrThrow(const ModelConfig& model_config) const;
  void validateModelConfigOrThrow() const;

  // constants
  static const std::string kInputTopic;
  static const std::string kOutputTopic;
  static const std::string kNoDetectionZoneTopic;
  static const std::string kNoDetectionZonePointsTopic;
  static const std::string kDetectionAreaTopic;
  static const std::string kModelBoundsTopic;
  static const std::map<uint8_t, std::vector<std::string>> kPossibleClassNames;
  static constexpr std::size_t kExpectedVarianceSize = 12;
  static constexpr int64_t kMinSensorId = 0;
  static constexpr int64_t kMaxSensorId = 100000;
  static constexpr int64_t kSensorIdStep = 1;
  static constexpr double kMinClassScoreThreshold = 0.0;
  static constexpr double kMaxClassScoreThreshold = 1.0;
  static constexpr double kMaxTritonClientTimeoutS = 300.0;
  static constexpr double kMinDetectionAreaRadius = 0.0;
  static constexpr double kMaxDetectionAreaRadius = 1000.0;
  static constexpr double kMinDetectionAreaBearingDeg = -360.0;
  static constexpr double kMaxDetectionAreaBearingDeg = 360.0;
  static constexpr double kMinDetectionAreaFovDeg = 0.0;
  static constexpr double kMaxDetectionAreaFovDeg = 360.0;
  static constexpr int64_t kMinDetectionAreaNumSegments = 3;
  static constexpr int64_t kMaxDetectionAreaNumSegments = 2048;

  // other member variables
  rclcpp::TimerBase::SharedPtr setup_timer_;
  std::unique_ptr<triton_cpp::TritonInterface> triton_interface_;

  // dynamic parameter callback
  OnSetParametersCallbackHandle::SharedPtr parameters_callback_;
  std::vector<std::tuple<std::string, std::function<void(const rclcpp::Parameter&)>>> auto_reconfigurable_params_;

  // Keep the transport factory alive for the full node lifetime so plugin loaders
  // outlive transport publishers/subscribers created from it.
  std::unique_ptr<point_cloud_transport::PointCloudTransport> point_cloud_transport_;

  // publisher and subscriber
  std::shared_ptr<point_cloud_transport::Subscriber> subscriber_;
  rclcpp::Publisher<perception_msgs::msg::ObjectList>::SharedPtr publisher_;
  rclcpp::Publisher<geometry_msgs::msg::PolygonStamped>::SharedPtr no_detection_zone_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PolygonStamped>::SharedPtr detection_area_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PolygonStamped>::SharedPtr model_bounds_pub_;
  std::mutex publishers_mutex_;

  // publisher for raw points inside the no-detection zone
  std::shared_ptr<point_cloud_transport::Publisher> no_detection_zone_points_publisher_;

  // transform listener and transform buffer
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;

 private:
  Params params_;
  std::string model_manifest_path_;
  ModelConfig model_config_;
  std::mutex model_mutex_;

  std::unique_ptr<Model> detection_model_;
  pcod_common::NmsConfig nms_config_;
  std::atomic<bool> model_ready_{false};
  std::atomic<bool> publishers_update_pending_{false};
};

}  // namespace point_cloud_object_detection
