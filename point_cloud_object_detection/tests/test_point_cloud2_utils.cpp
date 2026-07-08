// Copyright Institute for Automotive Engineering (ika), RWTH Aachen University
// SPDX-License-Identifier: Apache-2.0

#include "point_cloud_object_detection/PointCloud2Utils.hpp"

#include <cmath>
#include <cstring>
#include <stdexcept>

namespace {

template <typename T>
void writeScalar(std::vector<std::uint8_t>& data, std::size_t offset, T value) {
  if (offset + sizeof(T) > data.size()) {
    throw std::runtime_error("test write would exceed buffer size");
  }
  std::memcpy(&data[offset], &value, sizeof(T));
}

void expect(bool condition) {
  if (!condition) {
    throw std::runtime_error("unexpected PointCloud2 utility test value");
  }
}

template <typename T>
T byteSwapForTest(T value);

template <>
std::uint16_t byteSwapForTest(std::uint16_t value) {
  return point_cloud_object_detection::byteSwap16(value);
}

template <>
std::uint32_t byteSwapForTest(std::uint32_t value) {
  return point_cloud_object_detection::byteSwap32(value);
}

template <typename T>
void writeScalarSwapped(std::vector<std::uint8_t>& data, std::size_t offset, T value) {
  writeScalar(data, offset, byteSwapForTest(value));
}

void writeFloat32(std::vector<std::uint8_t>& data, std::size_t offset, float value, bool swapped = false) {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  if (swapped) {
    bits = point_cloud_object_detection::byteSwap32(bits);
  }
  writeScalar(data, offset, bits);
}

void writeFloat64(std::vector<std::uint8_t>& data, std::size_t offset, double value, bool swapped = false) {
  std::uint64_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  if (swapped) {
    bits = point_cloud_object_detection::byteSwap64(bits);
  }
  writeScalar(data, offset, bits);
}

}  // namespace

int main() {
  using point_cloud_object_detection::readPointCloud2PointRecord;
  using point_cloud_object_detection::readPointFieldAsFloat;
  using sensor_msgs::msg::PointField;

  sensor_msgs::msg::PointCloud2 msg;
  msg.width = 2;
  msg.height = 2;
  msg.point_step = 20;
  msg.row_step = 48;
  msg.data.resize(static_cast<std::size_t>(msg.row_step) * msg.height, 0U);

  const std::size_t second_row_first_col = static_cast<std::size_t>(msg.row_step);
  writeFloat32(msg.data, second_row_first_col + 0U, 1.25F);
  writeFloat32(msg.data, second_row_first_col + 4U, -2.5F);
  writeFloat32(msg.data, second_row_first_col + 8U, 3.75F);
  writeScalar<std::uint16_t>(msg.data, second_row_first_col + 12U, 42U);

  const auto point = readPointCloud2PointRecord(msg, 2U, 0U, 4U, 8U, 12U, PointField::UINT16, false);
  expect(std::abs(point.x - 1.25F) < 1e-6F);
  expect(std::abs(point.y + 2.5F) < 1e-6F);
  expect(std::abs(point.z - 3.75F) < 1e-6F);
  expect(std::abs(point.feature - 42.0F) < 1e-6F);

  std::vector<std::uint8_t> field_data(8U, 0U);
  writeScalar<std::uint8_t>(field_data, 0U, 7U);
  writeScalar<std::uint8_t>(field_data, 1U, static_cast<std::uint8_t>(-3));
  writeScalar<std::uint16_t>(field_data, 2U, 65530U);
  writeScalarSwapped<std::uint16_t>(field_data, 4U, static_cast<std::uint16_t>(static_cast<std::int16_t>(-12)));
  expect(std::abs(readPointFieldAsFloat(field_data, 0U, PointField::UINT8, false) - 7.0F) < 1e-6F);
  expect(std::abs(readPointFieldAsFloat(field_data, 1U, PointField::INT8, false) + 3.0F) < 1e-6F);
  expect(std::abs(readPointFieldAsFloat(field_data, 2U, PointField::UINT16, false) - 65530.0F) < 1e-6F);
  expect(std::abs(readPointFieldAsFloat(field_data, 4U, PointField::INT16, true) + 12.0F) < 1e-6F);

  field_data.assign(8U, 0U);
  writeScalar<std::uint32_t>(field_data, 0U, 123456U);
  writeScalarSwapped<std::uint32_t>(field_data, 4U, static_cast<std::uint32_t>(static_cast<std::int32_t>(-456)));
  expect(std::abs(readPointFieldAsFloat(field_data, 0U, PointField::UINT32, false) - 123456.0F) < 1e-3F);
  expect(std::abs(readPointFieldAsFloat(field_data, 4U, PointField::INT32, true) + 456.0F) < 1e-6F);

  field_data.assign(8U, 0U);
  writeFloat32(field_data, 0U, -9.5F, true);
  writeFloat64(field_data, 0U, 11.25, false);
  expect(std::abs(readPointFieldAsFloat(field_data, 0U, PointField::FLOAT64, false) - 11.25F) < 1e-6F);
  writeFloat32(field_data, 0U, -9.5F, true);
  expect(std::abs(readPointFieldAsFloat(field_data, 0U, PointField::FLOAT32, true) + 9.5F) < 1e-6F);

  bool threw = false;
  try {
    (void)readPointFieldAsFloat(field_data, 6U, PointField::FLOAT32, false);
  } catch (const std::out_of_range&) {
    threw = true;
  }
  expect(threw);

  return 0;
}
