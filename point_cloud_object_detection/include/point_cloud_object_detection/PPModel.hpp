#ifndef POINT_CLOUD_OBJECT_DETECTION__PP_MODEL_HPP_
#define POINT_CLOUD_OBJECT_DETECTION__PP_MODEL_HPP_

#include <string>
#include <vector>

#include "Definitions.hpp"
#include "Model.hpp"

namespace point_cloud_object_detection {

class PPModel : public Model {
 public:
  PPModel(triton_cpp::TritonInterface &triton_interface, ModelConfig &model_config, Params params);

  const std::string SAVED_MODEL_INPUT_NAME_PILLARS = "pillars/input";
  const std::string SAVED_MODEL_INPUT_NAME_INDICES = "pillars/indices";

  const std::string SAVED_MODEL_OUTPUT_NAME_OCCUPANCY = "occupancy/conv2d/Sigmoid";
  const std::string SAVED_MODEL_OUTPUT_NAME_LOCATION = "loc/reshape/Reshape";
  const std::string SAVED_MODEL_OUTPUT_NAME_SIZE = "size/reshape/Reshape";
  const std::string SAVED_MODEL_OUTPUT_NAME_ANGLE = "angle/conv2d/BiasAdd";
  const std::string SAVED_MODEL_OUTPUT_NAME_HEADING = "heading/conv2d/Sigmoid";
  const std::string SAVED_MODEL_OUTPUT_NAME_CLASS = "clf/reshape/Reshape";
  const std::string SAVED_MODEL_OUTPUT_NAME_VELOCITY = "vel/reshape/Reshape";

  virtual ~PPModel() = default;       // Any method virtual -> destructor virtual
  PPModel(const PPModel &) = delete;  // Rule of five
  PPModel &operator=(const PPModel &) = delete;
  PPModel(PPModel &&) = delete;
  PPModel &operator=(PPModel &&) = delete;

 protected:
  virtual void setupModelInput(const PointCloud &point_cloud) override;
  virtual std::vector<BoundingBox> modelOutputToBoxes() override;

 private:
  ModelConfig &model_config_;
  Params params_;

  const std::string input_name_pillars_;
  const std::string input_name_indices_;

  const std::string output_name_occupancy_;
  const std::string output_name_location_;
  const std::string output_name_size_;
  const std::string output_name_angle_;
  const std::string output_name_heading_;
  const std::string output_name_class_;
  const std::string output_name_velocity_;
};

}  // namespace point_cloud_object_detection

#endif  // POINT_CLOUD_OBJECT_DETECTION__PP_MODEL_HPP_