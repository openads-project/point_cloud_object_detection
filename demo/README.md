# Demo

This directory contains a self-contained demo setup for the `point_cloud_object_detection` node. It replays the [NVIDIA PhysicalAI-Autonomous-Vehicles Dataset](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles) with the help of the [autonomy_datasets](https://github.com/thinking-cars/autonomy_datasets) ROS package.

## Components

- [`docker-compose.yml`](./docker-compose.yml): Configuration of the demo.
- [`datasets/`](./datasets): The [NVIDIA PhysicalAI-Autonomous-Vehicles Dataset](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles) is converted into Rosbags which are then stored in this directory.
- [`models/`](./models): The detection model used for inference in the demo.
- [`point_cloud_object_detection.demo.params.yml`](./point_cloud_object_detection.demo.params.yml): Runtime parameters for the detection node used by the demo.
- [`rviz/config.rviz`](./rviz/config.rviz): RViz configuration for the demo setup.

## Services

- `triton-server`: Serves the model repository from [`./models`](./models).
- `point-cloud-object-detection`: Runs the packaged detection node against the Triton server.
- `rviz`: Visualizes `/demo/points`, `/demo/objects`, `/demo/detection_area`, `/demo/no_detection_zone`, `/demo/no_detection_zone_points`, and `/demo/model_bounds`.
- `ros-parameter-gui`: Starts `rqt_reconfigure` for interactive parameter changes.

## Run

1. Register at [HuggingFace](https://huggingface.co/join) and create a read-only [HuggingFace Access Token](https://huggingface.co/settings/tokens/new?tokenType=read).
2. Store the token in an `.env` file in the demo directory:

    ```bash
    echo "HF_TOKEN=your_token_here" > .env
    ```

3. Accept the terms and conditions for the dataset on [HuggingFace](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles).


4. Allow local Docker containers to connect to the X server for RViz visualization:

    ```bash
    xhost +local:
    ```

5. Start the demo from the demo directory (takes some time on the first run to pull the images and download the dataset): 
    ```bash
    docker compose up -d
    ```

6. Stop the demo from the demo directory once you're done:
    ```bash
    docker compose down
    ``` 

7.  Disable the connection to the X server after you're done with the demo:

    ```bash
    xhost -local:
    ```

## Interacting Through `rqt`

The `ros-parameter-gui` service starts `rqt` with the `rqt_reconfigure` plugin. You may use it to inspect and adjust the running parameters of the `point_cloud_object_detection` node while the demo is active.
