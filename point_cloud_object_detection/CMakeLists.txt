cmake_minimum_required(VERSION 3.8)
project(point_cloud_object_detection LANGUAGES C CXX)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

option(REQUIRE_CUDA "Fail build if CUDA is not found" OFF)

set(CUDA_AVAILABLE_MARKER "/opt/ros-cmake-cache/cuda-available")
set(CUDA_UNAVAILABLE_MARKER "/opt/ros-cmake-cache/cuda-unavailable")
if(EXISTS "${CUDA_UNAVAILABLE_MARKER}")
  message(STATUS "CUDA marked as unavailable. Skipping CUDA detection.")
  set(CUDA_EXPLICITLY_UNAVAILABLE TRUE)
else()
  set(CUDA_EXPLICITLY_UNAVAILABLE FALSE)
endif()

if(NOT CUDA_EXPLICITLY_UNAVAILABLE AND NOT DEFINED CMAKE_CUDA_COMPILER)
  if(EXISTS "/usr/local/cuda/bin/nvcc")
    set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc" CACHE FILEPATH "CUDA compiler" FORCE)
  elseif(EXISTS "/usr/local/cuda-12.8/bin/nvcc")
    set(CMAKE_CUDA_COMPILER "/usr/local/cuda-12.8/bin/nvcc" CACHE FILEPATH "CUDA compiler" FORCE)
  endif()
endif()

include(CheckLanguage)
if(NOT CUDA_EXPLICITLY_UNAVAILABLE)
  check_language(CUDA)
endif()

if(DEFINED CMAKE_CUDA_COMPILER AND NOT EXISTS "${CMAKE_CUDA_COMPILER}")
  if(CUDA_EXPLICITLY_UNAVAILABLE)
    message(STATUS "Ignoring CMAKE_CUDA_COMPILER on platform without CUDA support")
  elseif(REQUIRE_CUDA)
    message(FATAL_ERROR "REQUIRE_CUDA is ON and CMAKE_CUDA_COMPILER was set to '${CMAKE_CUDA_COMPILER}' but the compiler does not exist. Ensure CUDA Toolkit is installed.")
  else()
    message(STATUS "Specified CUDA compiler not found: ${CMAKE_CUDA_COMPILER}")
  endif()
  unset(CMAKE_CUDA_COMPILER)
  unset(CMAKE_CUDA_COMPILER CACHE)
endif()

if(CMAKE_CUDA_COMPILER)
  option(ENABLE_CUDA "Enable CUDA acceleration" ON)
else()
  option(ENABLE_CUDA "Enable CUDA acceleration" OFF)
endif()

if(EXISTS "${CUDA_AVAILABLE_MARKER}" AND NOT ENABLE_CUDA AND NOT CUDA_EXPLICITLY_UNAVAILABLE)
  message(STATUS "CUDA availability marker present but no CUDA compiler was detected")
endif()

if(ENABLE_CUDA)
  if(NOT CMAKE_CUDA_COMPILER)
    if(REQUIRE_CUDA)
      message(FATAL_ERROR "CUDA requested but no CUDA compiler found")
    endif()
    message(WARNING "CUDA requested but no CUDA compiler found. Disabling CUDA support.")
    set(ENABLE_CUDA OFF)
  else()
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    add_compile_definitions(ENABLE_CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
    message(STATUS "CUDA support enabled")
    message(STATUS "CUDA Toolkit version: ${CUDAToolkit_VERSION}")
  endif()
else()
  message(STATUS "CUDA support disabled (no CUDA compiler detected or ENABLE_CUDA=OFF)")
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(point_cloud_transport REQUIRED)
find_package(perception_msgs REQUIRED)
find_package(perception_msgs_utils REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
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
set(PCOD_COMMON_ENABLE_CUDA ${ENABLE_CUDA} CACHE BOOL "Enable optional CUDA components for pcod-common" FORCE)
add_subdirectory(pcod-common)
if(TARGET pcod_common)
  set_target_properties(pcod_common PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

add_library(
  ${PROJECT_NAME}_lib SHARED
  src/Model.cpp
  src/PBODModel.cpp
  src/PointCloudObjectDetection.cpp
  src/Utils.cpp
)

target_include_directories(
  ${PROJECT_NAME}_lib PUBLIC
                        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                        $<INSTALL_INTERFACE:include>
)
target_compile_features(
  ${PROJECT_NAME}_lib PUBLIC c_std_99 cxx_std_17)

ament_target_dependencies(
  ${PROJECT_NAME}_lib
  ament_index_cpp
  pcl_conversions
  pcl_ros
  perception_msgs
  perception_msgs_utils
  point_cloud_transport
  geometry_msgs
  nav_msgs
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

set(_triton_runtime_paths "/opt/tritonclient/lib;/opt/tritonclient/tritonserver/clients/lib")
set_target_properties(${PROJECT_NAME}_lib PROPERTIES
  BUILD_RPATH "${_triton_runtime_paths}"
  INSTALL_RPATH "${_triton_runtime_paths}"
  INSTALL_RPATH_USE_LINK_PATH TRUE
)
set_target_properties(${PROJECT_NAME} PROPERTIES
  BUILD_RPATH "${_triton_runtime_paths}"
  INSTALL_RPATH "${_triton_runtime_paths}"
  INSTALL_RPATH_USE_LINK_PATH TRUE
)

install(TARGETS ${PROJECT_NAME}_lib
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(DIRECTORY launch DESTINATION share/${PROJECT_NAME})
install(DIRECTORY config DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_clang_format_CONFIG_FILE ${CMAKE_CURRENT_SOURCE_DIR}/../.vscode/format/.clang-format)
  set(ament_cmake_clang_tidy_CONFIG_FILE ${CMAKE_CURRENT_SOURCE_DIR}/../.vscode/lint/.clang-tidy)
  set(ament_cmake_flake8_CONFIG_FILE ${CMAKE_CURRENT_SOURCE_DIR}/../.vscode/lint/ament_flake8.ini)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
