# point_cloud_object_detection

<p align="center">
  <a href="https://github.com/openads-project"><img src="https://img.shields.io/badge/OpenADS-f5ff01"/></a>
  <a href="https://www.ros.org"><img src="https://img.shields.io/badge/ROS 2-jazzy-22314e"/></a>
  <a href="https://github.com/openads-project/point_cloud_object_detection/releases/latest"><img src="https://img.shields.io/github/v/release/openads-project/point_cloud_object_detection"/></a>
  <a href="https://github.com/openads-project/point_cloud_object_detection/blob/main/LICENSE"><img src="https://img.shields.io/github/license/openads-project/point_cloud_object_detection"/></a>
  <br>
  <a href="https://github.com/openads-project/point_cloud_object_detection/actions/workflows/docker-ros.yml"><img src="https://github.com/openads-project/point_cloud_object_detection/actions/workflows/docker-ros.yml/badge.svg"/></a>
  <a href="https://github.com/openads-project/point_cloud_object_detection/actions/workflows/compose-oci.yml"><img src="https://github.com/openads-project/point_cloud_object_detection/actions/workflows/compose-oci.yml/badge.svg"/></a>
  <a href="https://openads-project.github.io/point_cloud_object_detection"><img src="https://github.com/openads-project/point_cloud_object_detection/actions/workflows/docs.yml/badge.svg"/></a>
  <a href="https://github.com/openads-project/point_cloud_object_detection/actions/workflows/consistency.yml"><img src="https://github.com/openads-project/point_cloud_object_detection/actions/workflows/consistency.yml/badge.svg"/></a>
</p>

This repository provides a ROS 2 point cloud object detection node for automated driving perception stacks. The node subscribes to a `sensor_msgs/msg/PointCloud2`, sends preprocessed point data to a Triton-served detection model, and publishes detected objects as `perception_msgs/msg/ObjectList`.

The detector itself does not host the neural network. A [Triton Inference Server](https://github.com/triton-inference-server/server) with a compatible exported model repository must be available at runtime.

<p align="center">
  <strong>🚀 <a href="#-quick-start">Quick Start</a></strong> • <strong>💻 <a href="#-development">Development</a></strong> • <strong>📝 <a href="#-documentation">Documentation</a></strong>
</p>


> [!IMPORTANT]
> This repository is part of [***OpenADS***](https://github.com/openads-project), the *Open Automated Driving Systems* project. *OpenADS* and its modules have been initiated and are currently being maintained by the [**Institute for Automotive Engineering (ika) at RWTH Aachen University**](https://www.ika.rwth-aachen.de/de/).


## 🚀 Quick Start

The [`demo`](demo) provides an example setup for the `point_cloud_object_detection` node. It can replay the [NVIDIA PhysicalAI-Autonomous-Vehicles Dataset](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles) with the help of the [autonomy_datasets](https://github.com/thinking-cars/autonomy_datasets) ROS package. Alternatively, it can replay local [PCD files](demo/pcds/).

1. Register at [HuggingFace](https://huggingface.co/join) and create a read-only [HuggingFace Access Token](https://huggingface.co/settings/tokens/new?tokenType=read).
2. Store the token in an `.env` file in the [demo](/demo) directory:

    ```bash
    echo "HF_TOKEN=your_token_here" > .env
    ```

3. Accept the terms and conditions for the dataset on [HuggingFace](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles).


4. Allow local Docker containers to connect to the X server for RViz visualization:

    ```bash
    xhost +local:
    ```

5. Start one of the replay profiles from the demo directory.

    ```bash
    cd demo
    docker compose --profile nvidia up -d --remove-orphans
    ```

   The `nvidia` profile iteratively downloads parts of the dataset, converts them to ROS bag files and replay the latest bag. Downloads are stopped once the demo is stopped. Downloading is resumed when the demo is resumed.
   
   If you want to replay and process provided PCD files instead, run:

    ```bash
    docker compose --profile pcds up -d --remove-orphans
    ```

6. Stop the demo from the demo directory once you're done:

    ```bash
    docker compose down --remove-orphans
    ``` 

7.  Disable the connection to the X server after you're done with the demo:

    ```bash
    xhost -local:
    ```

#### Interacting Through `rqt`

The `ros-parameter-gui` service starts `rqt` with the `rqt_reconfigure` plugin. You may use it to inspect and adjust the running parameters of the `point_cloud_object_detection` node while the demo is active.

## 💻 Development

### Set up Development Environment

1. Clone the repository.
    ```bash
    git clone https://gitlab.ika.rwth-aachen.de/fb-fi/its-modules/perception/point_cloud_object_detection.git
    ```
1. Initialize the [`.openads-dev-environment`](https://github.com/openads-project/openads-dev-environment) submodule containing development environment configuration.
    ```bash
    cd point_cloud_object_detection
    git submodule update --init --recursive
    ```
1. Open the repository in [Visual Studio Code](https://code.visualstudio.com).
    ```bash
    code .
    ```
1. Install the recommended VS Code extensions.
    > *Ctrl+Shift+P / Extensions: Show Recommended Extensions / Install Workspace Recommended Extensions (Cloud Download Icon)*
1. Reopen the repository in a [Dev Container](https://code.visualstudio.com/docs/devcontainers/containers).
    > *Ctrl+Shift+P / Dev Containers: Rebuild and Reopen in Container*

### Build

> *Ctrl+Shift+B*

```bash
colcon build
```

### Run Tests

> *Ctrl+Shift+P / Tasks: Run Test Task*

```bash
colcon build --cmake-args -DCMAKE_EXPORT_COMPILE_COMMANDS=1
colcon test
colcon test-result --verbose
```


## 📝 Documentation

Package and node interfaces are documented in the respective package READMEs listed below. Implementation details are found in the [Source Code Documentation](https://fb-fi.pages.ika.rwth-aachen.de/its-modules/perception/point_cloud_object_detection).

| Package | Description |
| --- | --- |
| [point_cloud_object_detection](point_cloud_object_detection/README.md) | Detects objects in point clouds |

## ⚖️ Licensing

The source code in this repository is licensed under Apache-2.0, see [LICENSE](LICENSE). Container images provided by this repository may contain third-party software shipped with their own license terms.

## 🙏 Acknowledgements

Development and maintenance of this repository are supported by the following projects. We acknowledge the funding of the respective institutions.

| Project | Funding Institution | Grant Number |
| --- | --- | --- |
| [AIGGREGATE](https://aiggregate.eu/) | 🇪🇺 European Union | 101202457 |
