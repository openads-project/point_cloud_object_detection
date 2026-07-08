// Copyright Institute for Automotive Engineering (ika), RWTH Aachen University
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/point_field.hpp>
#include <stdexcept>
#include <vector>

namespace point_cloud_object_detection {

/**
 * @brief Decoded point values read from one PointCloud2 record.
 */
struct PointCloud2PointRecord {
  float x;
  float y;
  float z;
  float feature;
};

/**
 * @brief Swap the byte order of a 16-bit scalar.
 */
inline std::uint16_t byteSwap16(std::uint16_t value) { return static_cast<std::uint16_t>((value >> 8) | (value << 8)); }

/**
 * @brief Swap the byte order of a 32-bit scalar.
 */
inline std::uint32_t byteSwap32(std::uint32_t value) {
  return ((value & 0x000000FFU) << 24) | ((value & 0x0000FF00U) << 8) | ((value & 0x00FF0000U) >> 8) |
         ((value & 0xFF000000U) >> 24);
}

/**
 * @brief Swap the byte order of a 64-bit scalar.
 */
inline std::uint64_t byteSwap64(std::uint64_t value) {
  return ((value & 0x00000000000000FFULL) << 56) | ((value & 0x000000000000FF00ULL) << 40) |
         ((value & 0x0000000000FF0000ULL) << 24) | ((value & 0x00000000FF000000ULL) << 8) |
         ((value & 0x000000FF00000000ULL) >> 8) | ((value & 0x0000FF0000000000ULL) >> 24) |
         ((value & 0x00FF000000000000ULL) >> 40) | ((value & 0xFF00000000000000ULL) >> 56);
}

/**
 * @brief Read a trivially copyable scalar from a byte buffer.
 */
template <typename T>
T readScalarAt(const std::vector<std::uint8_t>& data, std::size_t offset) {
  if (offset + sizeof(T) > data.size()) {
    throw std::out_of_range("PointCloud2 field offset exceeds data buffer");
  }
  T value{};
  std::memcpy(&value, &data[offset], sizeof(T));
  return value;
}

/**
 * @brief Read a PointCloud2 FLOAT32 field as host-order float.
 */
inline float readFloat32At(const std::vector<std::uint8_t>& data, std::size_t offset, bool needs_swap) {
  std::uint32_t bits = readScalarAt<std::uint32_t>(data, offset);
  if (needs_swap) {
    bits = byteSwap32(bits);
  }
  float out = 0.0F;
  std::memcpy(&out, &bits, sizeof(out));
  return out;
}

/**
 * @brief Read a PointCloud2 FLOAT64 field and convert it to float.
 */
inline float readFloat64AsFloatAt(const std::vector<std::uint8_t>& data, std::size_t offset, bool needs_swap) {
  std::uint64_t bits = readScalarAt<std::uint64_t>(data, offset);
  if (needs_swap) {
    bits = byteSwap64(bits);
  }
  double out = 0.0;
  std::memcpy(&out, &bits, sizeof(out));
  return static_cast<float>(out);
}

/**
 * @brief Read a supported PointCloud2 field datatype as float.
 */
inline float readPointFieldAsFloat(const std::vector<std::uint8_t>& data,
                                   std::size_t offset,
                                   std::uint8_t datatype,
                                   bool needs_swap) {
  using PF = sensor_msgs::msg::PointField;
  switch (datatype) {
    case PF::FLOAT32:
      return readFloat32At(data, offset, needs_swap);
    case PF::FLOAT64:
      return readFloat64AsFloatAt(data, offset, needs_swap);
    case PF::UINT16: {
      std::uint16_t value = readScalarAt<std::uint16_t>(data, offset);
      if (needs_swap) {
        value = byteSwap16(value);
      }
      return static_cast<float>(value);
    }
    case PF::UINT8:
      return static_cast<float>(readScalarAt<std::uint8_t>(data, offset));
    case PF::INT16: {
      std::uint16_t raw = readScalarAt<std::uint16_t>(data, offset);
      if (needs_swap) {
        raw = byteSwap16(raw);
      }
      return static_cast<float>(static_cast<std::int16_t>(raw));
    }
    case PF::INT8:
      return static_cast<float>(static_cast<std::int8_t>(readScalarAt<std::uint8_t>(data, offset)));
    case PF::UINT32: {
      std::uint32_t value = readScalarAt<std::uint32_t>(data, offset);
      if (needs_swap) {
        value = byteSwap32(value);
      }
      return static_cast<float>(value);
    }
    case PF::INT32: {
      std::uint32_t raw = readScalarAt<std::uint32_t>(data, offset);
      if (needs_swap) {
        raw = byteSwap32(raw);
      }
      return static_cast<float>(static_cast<std::int32_t>(raw));
    }
    default:
      throw std::runtime_error("Unsupported PointCloud2 datatype");
  }
}

/**
 * @brief Compute the byte offset of a point record in a PointCloud2 message.
 */
inline std::size_t pointCloud2PointOffset(const sensor_msgs::msg::PointCloud2& msg, std::size_t index) {
  const std::size_t width = msg.width;
  const std::size_t row = index / width;
  const std::size_t col = index % width;
  return row * static_cast<std::size_t>(msg.row_step) + col * static_cast<std::size_t>(msg.point_step);
}

/**
 * @brief Read XYZ and feature values from one PointCloud2 point record.
 */
inline PointCloud2PointRecord readPointCloud2PointRecord(const sensor_msgs::msg::PointCloud2& msg,
                                                         std::size_t index,
                                                         std::size_t x_offset,
                                                         std::size_t y_offset,
                                                         std::size_t z_offset,
                                                         std::size_t feature_offset,
                                                         std::uint8_t feature_datatype,
                                                         bool needs_swap) {
  const std::size_t point_offset = pointCloud2PointOffset(msg, index);
  return {readFloat32At(msg.data, point_offset + x_offset, needs_swap),
          readFloat32At(msg.data, point_offset + y_offset, needs_swap),
          readFloat32At(msg.data, point_offset + z_offset, needs_swap),
          readPointFieldAsFloat(msg.data, point_offset + feature_offset, feature_datatype, needs_swap)};
}

}  // namespace point_cloud_object_detection
