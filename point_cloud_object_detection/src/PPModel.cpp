#include "point_cloud_object_detection/PPModel.hpp"

namespace point_cloud_object_detection {
PPModel::PPModel(triton_cpp::TritonInterface& triton_interface, ModelConfig& model_config, Params params)
    : Model(triton_interface),
      model_config_{model_config},
      params_{params},
      input_name_pillars_{SAVED_MODEL_INPUT_NAME_PILLARS},
      input_name_indices_{SAVED_MODEL_INPUT_NAME_INDICES},
      output_name_occupancy_{SAVED_MODEL_OUTPUT_NAME_OCCUPANCY},
      output_name_location_{SAVED_MODEL_OUTPUT_NAME_LOCATION},
      output_name_size_{SAVED_MODEL_OUTPUT_NAME_SIZE},
      output_name_angle_{SAVED_MODEL_OUTPUT_NAME_ANGLE},
      output_name_heading_{SAVED_MODEL_OUTPUT_NAME_HEADING},
      output_name_class_{SAVED_MODEL_OUTPUT_NAME_CLASS},
      output_name_velocity_{SAVED_MODEL_OUTPUT_NAME_VELOCITY} {}

void PPModel::setupModelInput(const PointCloud& point_cloud) {
  auto pillar_map = triton_interface_.getInputTensor<float>(
      input_name_pillars_, model_config_.max_pillars, model_config_.max_points_per_pillar, model_config_.n_features);
  auto indices_map = triton_interface_.getInputTensor<int>(input_name_indices_, model_config_.max_pillars, 3);
  pillar_map.setZero();
  indices_map.setZero();

  // Clear filtered input points
  filtered_input_points_.clear();

  std::unordered_map<std::pair<uint32_t, uint32_t>, std::vector<PillarPoint>, IntPairHash> map;
  for (PointCloud::const_iterator point = point_cloud.begin(); point != point_cloud.end(); ++point) {
    // filter points to detection range
    if ((point->x < model_config_.x_min) || (point->x >= model_config_.x_max) || (point->y < model_config_.y_min) ||
        (point->y >= model_config_.y_max) || (point->z < model_config_.z_min) || (point->z >= model_config_.z_max)) {
      continue;
    }

    // Optional: filter points outside detection area (sector) in inference_frame
    if (model_config_.detection_area_remove_points_outside) {
      const double delta_x = static_cast<double>(point->x) - model_config_.detection_area_center_x;
      const double delta_y = static_cast<double>(point->y) - model_config_.detection_area_center_y;
      const double squared_distance_to_center = delta_x * delta_x + delta_y * delta_y;
      const double sector_radius = model_config_.detection_area_radius;
      if (squared_distance_to_center > sector_radius * sector_radius) {
        continue;
      }
      const double angle_to_center = std::atan2(delta_y, delta_x);
      const double sector_bearing_rad = model_config_.detection_area_bearing_deg * M_PI / 180.0;
      double angle_offset = angle_to_center - sector_bearing_rad;
      while (angle_offset > M_PI) angle_offset -= 2.0 * M_PI;
      while (angle_offset < -M_PI) angle_offset += 2.0 * M_PI;
      if (std::abs(angle_offset) > (model_config_.detection_area_fov_deg * M_PI / 180.0) * 0.5 + 1e-9) {
        continue;
      }
    }
    auto x_index = static_cast<uint32_t>(std::floor((point->x - model_config_.x_min) / model_config_.delta_x));
    auto y_index = static_cast<uint32_t>(std::floor((point->y - model_config_.y_min) / model_config_.delta_y));

    float intensity = point->intensity / model_config_.intensity_threshold;
    intensity = std::min(intensity, 1.F);
    intensity = std::max(intensity, 0.F);

    map[{x_index, y_index}].emplace_back(PillarPoint{point->x, point->y, point->z, intensity, 0.F, 0.F, 0.F});
    // Store this point as it passed the filtering
    filtered_input_points_.push_back(*point);
  }

  // iterate over all pillFars
  int pillar_id = 0;
  for (auto& pair : map) {
    // check max_pillars
    if (pillar_id >= model_config_.max_pillars) {
      break;
    }

    // calculate pillar mean
    float x_mean = 0;
    float y_mean = 0;
    float z_mean = 0;
    for (const auto& p : pair.second) {
      x_mean += p.x;
      y_mean += p.y;
      z_mean += p.z;
    }
    x_mean /= pair.second.size();
    y_mean /= pair.second.size();
    z_mean /= pair.second.size();

    // calculate deviation to mean
    for (auto& p : pair.second) {
      p.xc = p.x - x_mean;
      p.yc = p.y - y_mean;
      p.zc = p.z - z_mean;
    }

    // save indices in map
    const auto& x_index = pair.first.first;
    const auto& y_index = pair.first.second;
    indices_map(pillar_id, 1) = x_index;
    indices_map(pillar_id, 2) = y_index;

    // iterate over all points in a pillar
    int point_id = 0;
    for (const auto& p : pair.second) {
      // check max_points_per_pillar
      if (point_id >= model_config_.max_points_per_pillar) {
        break;
      }

      // build 9 dimensional network input from here (chapter 2.1 https://arxiv.org/pdf/1812.05784.pdf)
      pillar_map(pillar_id, point_id, 0) = p.x;
      pillar_map(pillar_id, point_id, 1) = p.y;
      pillar_map(pillar_id, point_id, 2) = p.z;
      pillar_map(pillar_id, point_id, 3) = p.intensity;
      // subscript c refers to the distance to the pillar mean
      pillar_map(pillar_id, point_id, 4) = p.xc;
      pillar_map(pillar_id, point_id, 5) = p.yc;
      pillar_map(pillar_id, point_id, 6) = p.zc;
      // subscript p refers to the offset to the pillar center
      pillar_map(pillar_id, point_id, 7) = p.x - (x_index * model_config_.delta_x + model_config_.x_min);
      pillar_map(pillar_id, point_id, 8) = p.y - (y_index * model_config_.delta_y + model_config_.y_min);

      ++point_id;
    }

    ++pillar_id;
  }
}

std::vector<BoundingBox> PPModel::modelOutputToBoxes() {
  int x_grid_size = model_config_.x_grid_size / model_config_.downscaling;
  int y_grid_size = model_config_.y_grid_size / model_config_.downscaling;

  auto occupancy = triton_interface_.getOutputTensor<float>(output_name_occupancy_, x_grid_size, y_grid_size,
                                                            model_config_.anchor_boxes.size());
  auto location = triton_interface_.getOutputTensor<float>(output_name_location_, x_grid_size, y_grid_size,
                                                           model_config_.anchor_boxes.size(), 3);
  auto size = triton_interface_.getOutputTensor<float>(output_name_size_, x_grid_size, y_grid_size,
                                                       model_config_.anchor_boxes.size(), 3);
  auto angle = triton_interface_.getOutputTensor<float>(output_name_angle_, x_grid_size, y_grid_size,
                                                        model_config_.anchor_boxes.size());
  auto classification = triton_interface_.getOutputTensor<float>(output_name_class_, x_grid_size, y_grid_size,
                                                                 model_config_.anchor_boxes.size(),
                                                                 model_config_.predicted_class_names.size());
  // Heading is not used for now
  // Eigen::TensorMap<
  //   Eigen::Tensor<const float, 4, Eigen::RowMajor, int64_t>, Eigen::Aligned, Eigen::MakePointer>
  //   heading = model_output.at(output_name_heading_).tensor<float, 4>();
  triton_cpp::TensorType<const float, 4> velocity{
      classification};  // Initialize velocity with classes as no default constructor exists
  if (model_config_.with_velocity) {
    velocity =
        triton_interface_.getOutputTensor<float>(output_name_velocity_, model_config_.x_grid_size,
                                                 model_config_.y_grid_size, model_config_.anchor_boxes.size(), 2);
  }
  std::vector<BoundingBox> objects;
  // iterate over grid
  for (int x_index = 0; x_index < occupancy.dimension(0); ++x_index) {
    for (int y_index = 0; y_index < occupancy.dimension(1); ++y_index) {
      // calculate the mean probability over all anchors
      float mean_probability = 0.0;
      for (int j = 0; j < occupancy.dimension(2); ++j) {
        mean_probability += occupancy(x_index, y_index, j);
      }
      mean_probability /= occupancy.dimension(2);

      // iterate over all anchors
      for (int anchor = 0; anchor < occupancy.dimension(2); ++anchor) {
        // check for minimum_tresh, just for runtime reasons
        if (occupancy(x_index, y_index, anchor) < model_config_.min_nms_score_threshold) {
          continue;
        }

        // initilize bounding_box
        BoundingBox bounding_box;

        // position calculation
        // The origin is basically placed in the middle of a cell although this is
        // technically not possibly. The grid requires an even number of cells
        // an the origin is therefore placed at the cross of the four center
        // cells. However, the network was trained to learn the offset from the
        // bottom right corner of the cell.
        float x_center = x_index * model_config_.delta_x * model_config_.downscaling + model_config_.x_min;
        float y_center = y_index * model_config_.delta_y * model_config_.downscaling + model_config_.y_min;

        // set position of bounding_box
        bounding_box.center[0] =
            x_center + location(x_index, y_index, anchor, 0) * model_config_.anchor_diagonals[anchor];
        bounding_box.center[1] =
            y_center + location(x_index, y_index, anchor, 1) * model_config_.anchor_diagonals[anchor];
        bounding_box.z = model_config_.anchor_boxes[anchor].z_center +
                         location(x_index, y_index, anchor, 2) * model_config_.anchor_boxes[anchor].height;

        // set size of bounding_box
        bounding_box.length = std::exp(size(x_index, y_index, anchor, 0)) * model_config_.anchor_boxes[anchor].length;
        bounding_box.width = std::exp(size(x_index, y_index, anchor, 1)) * model_config_.anchor_boxes[anchor].width;
        bounding_box.height = std::exp(size(x_index, y_index, anchor, 2)) * model_config_.anchor_boxes[anchor].height;

        // set yaw of bounding_box
        bounding_box.yaw =
            std::asin(clamp(angle(x_index, y_index, anchor), -1.0F, 1.0F)) + model_config_.anchor_boxes[anchor].yaw;

        // filter boxes containing nan values
        if (std::isnan(bounding_box.length) || std::isnan(bounding_box.width) || std::isnan(bounding_box.height) ||
            std::isnan(bounding_box.yaw)) {
          continue;
        }

        // build class_score vector
        std::vector<float> class_scores;
        for (int i = 0; i < classification.dimension(3); i++) {
          class_scores.push_back(classification(x_index, y_index, anchor, i));
        }

        // check if class_names has same dimension
        if (class_scores.size() != model_config_.predicted_class_names.size()) {
          std::cerr << "Detector Error: 'predicted_class_names' is required to have the same size than the "
                       "cls layer of the neural network"
                    << std::endl;
        }

        // set class_idx and score
        std::vector<float>::iterator it_max = std::max_element(class_scores.begin(), class_scores.end());
        int class_idx = std::distance(class_scores.begin(), it_max);

        float score = occupancy(x_index, y_index, anchor);

        // filter by specific detection thresholds.
        if (score < model_config_.nms_score_threshold[class_idx]) {
          continue;
        }

        // filter all anchors with lower probability than mean
        if (occupancy(x_index, y_index, anchor) < mean_probability) {
          continue;
        }

        // set classes and scores
        for (std::size_t i = 0; i < model_config_.predicted_class_names.size(); i++) {
          bounding_box.classification.push_back({i, classification(x_index, y_index, anchor, i)});
        }
        bounding_box.existence_probability = score;

        // set velocity
        if (model_config_.with_velocity) {
          bounding_box.has_velocity = true;
          bounding_box.v_x = velocity(x_index, y_index, anchor, 0);
          bounding_box.v_y = velocity(x_index, y_index, anchor, 1);
        }

        objects.push_back(bounding_box);
      }
    }
  }

  return objects;
}

}  // namespace point_cloud_object_detection
