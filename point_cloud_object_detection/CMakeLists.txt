cmake_minimum_required(VERSION 3.8)
project(point_cloud_object_detection)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
# uncomment the following section in order to fill in further dependencies
# manually. find_package(<dependency> REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(point_cloud_transport REQUIRED)
find_package(perception_msgs REQUIRED)
find_package(perception_msgs_utils REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_path(TRITON_CONFIG_DIR TritonConfig.cmake PATHS "../../upstream/triton_cpp" "/root/upstream_ws/src/triton_cpp"
                                               PATH_SUFFIXES "cmake")

list(APPEND CMAKE_PREFIX_PATH ${TRITON_CONFIG_DIR})
find_package(triton_cpp REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_perception_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_sensor_msgs REQUIRED)

set(PCOD_COMMON_BUILD_TESTS OFF CACHE BOOL "Disable pcod-common tests in this workspace" FORCE)
add_subdirectory(pcod-common)
if(TARGET pcod_common)
  set_target_properties(pcod_common PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

add_library(
  ${PROJECT_NAME}_lib SHARED
  src/PointCloudObjectDetection.cpp
  src/Utils.cpp
  src/Model.cpp
  src/PBODModel.cpp
  #src/TPODModel.cpp
)

target_include_directories(
  ${PROJECT_NAME}_lib PUBLIC
                        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                        $<INSTALL_INTERFACE:include>
)
# target_include_directories(${PROJECT_NAME}_lib BEFORE
#                         PUBLIC ${tensorflow_cpp_INCLUDE_DIRS})
target_compile_features(
  ${PROJECT_NAME}_lib PUBLIC c_std_99 cxx_std_17) # Require C99 and C++17

ament_target_dependencies(
  ${PROJECT_NAME}_lib
  ament_index_cpp
  pcl_conversions
  pcl_ros
  perception_msgs
  perception_msgs_utils
  point_cloud_transport
  geometry_msgs
  rclcpp
  rclcpp_components
  sensor_msgs
  std_msgs
  triton_cpp
  tf2
  tf2_perception_msgs
  tf2_ros
  tf2_sensor_msgs
)

target_link_libraries(${PROJECT_NAME}_lib pcod_common)

rclcpp_components_register_node(${PROJECT_NAME}_lib
  PLUGIN "point_cloud_object_detection::PointCloudObjectDetection"
  EXECUTABLE ${PROJECT_NAME})

install(TARGETS ${PROJECT_NAME}_lib
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(DIRECTORY launch DESTINATION share/${PROJECT_NAME})

install(DIRECTORY config DESTINATION share/${PROJECT_NAME})

install(DIRECTORY model_manifests DESTINATION share/${PROJECT_NAME})

install(DIRECTORY scripts DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)

  set(ament_cmake_clang_format_CONFIG_FILE $ENV{WORKSPACE}/src/target/.vscode/format/.clang-format)
  set(ament_cmake_clang_tidy_CONFIG_FILE $ENV{WORKSPACE}/src/target/.vscode/lint/.clang-tidy)
  set(ament_cmake_flake8_CONFIG_FILE $ENV{WORKSPACE}/src/target/.vscode/lint/ament_flake8.ini)

  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
